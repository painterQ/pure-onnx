package openclip

import (
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"strings"
	"time"
)

const (
	// DefaultBootstrapRepoID is the default Hugging Face repository for OpenCLIP ONNX artifacts.
	DefaultBootstrapRepoID = "amikos/openclip-vit-b-32-laion2b-s34b-b79k-onnx"
	// DefaultBootstrapRevision is the default repository revision used by the bootstrapper.
	DefaultBootstrapRevision = "248a2ed76a7189fc080e654e36930171331ef085"
	// DefaultBootstrapBaseURL is the default Hugging Face host used by the bootstrapper.
	DefaultBootstrapBaseURL = "https://huggingface.co"
)

const (
	// #nosec G101 -- deterministic artifact checksum, not credential material.
	defaultTextModelSHA256 = "252b86e0ef1fc95b22cfd52fbf647142727fdbecc152556ffe0fba0b10a80370"
	// #nosec G101 -- deterministic artifact checksum, not credential material.
	defaultVisionSHA256 = "7e14f76233d0c840c0621b1ef68f5877efe9357850782b1bbaf0c01693f73b43"
	// #nosec G101 -- deterministic artifact checksum, not credential material.
	defaultTokenizerSHA256 = "b556ac8c99757ffb677208af34bc8c6721572114111a6e0aaf5fa69ff0b8d842"
	// #nosec G101 -- deterministic artifact checksum, not credential material.
	defaultPreprocSHA256 = "910e70b3956ac9879ebc90b22fb3bc8a75b6a0677814500101a4c072bd7857bd"
)

const (
	defaultTextModelSize    int64 = 253973818
	defaultVisionModelSize  int64 = 351618799
	defaultTokenizerSize    int64 = 2224041
	defaultPreprocessorSize int64 = 316
)

const (
	textModelFileName   = "text_model.onnx"
	visionModelFileName = "vision_model.onnx"
	// #nosec G101 -- fixed artifact file name, not credential material.
	tokenizerFileName = "tokenizer.json"
	// #nosec G101 -- fixed artifact file name, not credential material.
	preprocessorFileName = "preprocessor_config.json"
)

const (
	defaultMaxDownloadBytes int64         = 1 << 30 // 1 GiB cap for unpinned/custom bundles.
	defaultLockWaitTimeout  time.Duration = 30 * time.Second
	defaultLockStaleAfter   time.Duration = 10 * time.Minute
	defaultHTTPTimeout      time.Duration = 60 * time.Second
)

var bootstrapAssetSpecs = []bootstrapAssetSpec{
	{fileName: textModelFileName},
	{fileName: visionModelFileName},
	{fileName: tokenizerFileName},
	{fileName: preprocessorFileName},
}

// ModelAssets describes a local OpenCLIP artifact bundle.
type ModelAssets struct {
	TextModelPath          string
	VisionModelPath        string
	TokenizerPath          string
	PreprocessorConfigPath string
}

// BootstrapOption customizes EnsureDefaultAssets behavior.
type BootstrapOption func(*bootstrapConfig) error

type bootstrapConfig struct {
	repoID             string
	revision           string
	baseURL            string
	cacheDir           string
	hfToken            string
	verifySHA          bool
	shaByFile          map[string]string
	expectedSizeByFile map[string]int64
	maxDownloadBytes   int64
	httpClient         *http.Client
}

type bootstrapAssetSpec struct {
	fileName string
}

type downloadStatusError struct {
	StatusCode int
	URL        string
	Body       string
}

func (e *downloadStatusError) Error() string {
	if e == nil {
		return "download status error: <nil>"
	}
	if e.Body == "" {
		return fmt.Sprintf("HTTP %d for %s", e.StatusCode, e.URL)
	}
	return fmt.Sprintf("HTTP %d for %s: %s", e.StatusCode, e.URL, e.Body)
}

type fileLock struct {
	path string
}

func (l *fileLock) Release() error {
	if l == nil {
		return nil
	}
	if err := os.Remove(l.path); err != nil && !errors.Is(err, os.ErrNotExist) {
		return err
	}
	return nil
}

// WithBootstrapCacheDir sets the local cache directory used for downloaded assets.
func WithBootstrapCacheDir(path string) BootstrapOption {
	return func(cfg *bootstrapConfig) error {
		if strings.TrimSpace(path) == "" {
			return fmt.Errorf("bootstrap cache directory cannot be empty")
		}
		cfg.cacheDir = path
		return nil
	}
}

// WithBootstrapRepoID sets the Hugging Face repo ID to fetch assets from.
func WithBootstrapRepoID(repoID string) BootstrapOption {
	return func(cfg *bootstrapConfig) error {
		if strings.TrimSpace(repoID) == "" {
			return fmt.Errorf("bootstrap repo ID cannot be empty")
		}
		cfg.repoID = repoID
		return nil
	}
}

// WithBootstrapRevision sets the Hugging Face revision to fetch assets from.
func WithBootstrapRevision(revision string) BootstrapOption {
	return func(cfg *bootstrapConfig) error {
		if strings.TrimSpace(revision) == "" {
			return fmt.Errorf("bootstrap revision cannot be empty")
		}
		cfg.revision = revision
		return nil
	}
}

// WithBootstrapToken sets an optional Hugging Face access token for downloads.
func WithBootstrapToken(token string) BootstrapOption {
	return func(cfg *bootstrapConfig) error {
		cfg.hfToken = strings.TrimSpace(token)
		return nil
	}
}

// WithBootstrapChecksum pins a SHA256 checksum for a required artifact.
func WithBootstrapChecksum(fileName string, checksum string) BootstrapOption {
	return func(cfg *bootstrapConfig) error {
		if err := validateAssetFileName(fileName); err != nil {
			return err
		}
		normalized := strings.ToLower(strings.TrimSpace(checksum))
		if err := validateSHA256(normalized); err != nil {
			return fmt.Errorf("invalid checksum for %s: %w", fileName, err)
		}
		cfg.shaByFile[fileName] = normalized
		return nil
	}
}

// WithBootstrapExpectedSize pins an exact byte size for a required artifact.
func WithBootstrapExpectedSize(fileName string, sizeBytes int64) BootstrapOption {
	return func(cfg *bootstrapConfig) error {
		if err := validateAssetFileName(fileName); err != nil {
			return err
		}
		if sizeBytes <= 0 {
			return fmt.Errorf("expected size for %s must be > 0, got %d", fileName, sizeBytes)
		}
		cfg.expectedSizeByFile[fileName] = sizeBytes
		return nil
	}
}

// WithBootstrapMaxDownloadBytes sets a hard per-file cap to protect disk usage.
func WithBootstrapMaxDownloadBytes(limit int64) BootstrapOption {
	return func(cfg *bootstrapConfig) error {
		if limit <= 0 {
			return fmt.Errorf("max download bytes must be > 0, got %d", limit)
		}
		cfg.maxDownloadBytes = limit
		return nil
	}
}

// WithoutBootstrapChecksumVerification disables checksum verification for downloaded assets.
func WithoutBootstrapChecksumVerification() BootstrapOption {
	return func(cfg *bootstrapConfig) error {
		cfg.verifySHA = false
		return nil
	}
}

// EnsureDefaultAssets ensures OpenCLIP assets are present locally and returns their paths.
//
// By default this fetches:
//   - repo: amikos/openclip-vit-b-32-laion2b-s34b-b79k-onnx
//   - revision: 248a2ed76a7189fc080e654e36930171331ef085
//   - files: text_model.onnx, vision_model.onnx, tokenizer.json, preprocessor_config.json
func EnsureDefaultAssets(opts ...BootstrapOption) (ModelAssets, error) {
	cfg, err := defaultBootstrapConfig()
	if err != nil {
		return ModelAssets{}, err
	}
	for _, opt := range opts {
		if err := opt(&cfg); err != nil {
			return ModelAssets{}, err
		}
	}
	applyDefaultBundleIntegrity(&cfg)
	if err := validateIntegrityConfig(cfg); err != nil {
		return ModelAssets{}, err
	}
	ensureHTTPClientSecurity(&cfg)
	return ensureModelAssets(cfg)
}

func defaultBootstrapConfig() (bootstrapConfig, error) {
	cacheDir := strings.TrimSpace(os.Getenv("ONNXRUNTIME_OPENCLIP_CACHE_DIR"))
	if cacheDir == "" {
		cacheDir = strings.TrimSpace(os.Getenv("ONNXRUNTIME_TEST_MODEL_CACHE_DIR"))
	}
	if cacheDir == "" {
		userCacheDir, err := os.UserCacheDir()
		if err != nil {
			return bootstrapConfig{}, fmt.Errorf("cannot determine user cache directory: %w", err)
		}
		cacheDir = filepath.Join(userCacheDir, "onnx-purego", "openclip")
	}

	return bootstrapConfig{
		repoID:             DefaultBootstrapRepoID,
		revision:           DefaultBootstrapRevision,
		baseURL:            DefaultBootstrapBaseURL,
		cacheDir:           cacheDir,
		hfToken:            strings.TrimSpace(os.Getenv("HF_TOKEN")),
		verifySHA:          true,
		shaByFile:          map[string]string{},
		expectedSizeByFile: map[string]int64{},
		maxDownloadBytes:   defaultMaxDownloadBytes,
		httpClient: &http.Client{
			Timeout: defaultHTTPTimeout,
		},
	}, nil
}

func applyDefaultBundleIntegrity(cfg *bootstrapConfig) {
	if cfg == nil || !cfg.verifySHA {
		return
	}
	if cfg.repoID != DefaultBootstrapRepoID || cfg.revision != DefaultBootstrapRevision {
		return
	}

	cfg.shaByFile[textModelFileName] = defaultTextModelSHA256
	cfg.shaByFile[visionModelFileName] = defaultVisionSHA256
	cfg.shaByFile[tokenizerFileName] = defaultTokenizerSHA256
	cfg.shaByFile[preprocessorFileName] = defaultPreprocSHA256

	cfg.expectedSizeByFile[textModelFileName] = defaultTextModelSize
	cfg.expectedSizeByFile[visionModelFileName] = defaultVisionModelSize
	cfg.expectedSizeByFile[tokenizerFileName] = defaultTokenizerSize
	cfg.expectedSizeByFile[preprocessorFileName] = defaultPreprocessorSize
}

func validateIntegrityConfig(cfg bootstrapConfig) error {
	if !cfg.verifySHA {
		return nil
	}
	for _, spec := range bootstrapAssetSpecs {
		checksum := strings.TrimSpace(cfg.shaByFile[spec.fileName])
		if checksum == "" {
			return fmt.Errorf(
				"missing checksum for required asset %q. "+
					"Provide WithBootstrapChecksum(...) for every required file, or explicitly disable verification with WithoutBootstrapChecksumVerification()",
				spec.fileName,
			)
		}
		if err := validateSHA256(checksum); err != nil {
			return fmt.Errorf("invalid checksum for %s: %w", spec.fileName, err)
		}
	}
	return nil
}

func ensureHTTPClientSecurity(cfg *bootstrapConfig) {
	if cfg == nil {
		return
	}
	if cfg.httpClient == nil {
		cfg.httpClient = &http.Client{
			Timeout:       defaultHTTPTimeout,
			CheckRedirect: makeRedirectPolicy(cfg.baseURL),
		}
		return
	}
	if cfg.httpClient.Timeout <= 0 {
		cfg.httpClient.Timeout = defaultHTTPTimeout
	}
	if cfg.httpClient.CheckRedirect == nil {
		cfg.httpClient.CheckRedirect = makeRedirectPolicy(cfg.baseURL)
	}
}

func makeRedirectPolicy(baseURL string) func(req *http.Request, via []*http.Request) error {
	baseHost := ""
	if parsed, err := url.Parse(strings.TrimSpace(baseURL)); err == nil {
		baseHost = strings.ToLower(parsed.Hostname())
	}
	return func(req *http.Request, via []*http.Request) error {
		if len(via) == 0 {
			return nil
		}
		if len(via) >= 10 {
			return fmt.Errorf("redirect blocked: too many redirects (%d)", len(via))
		}
		previous := via[len(via)-1].URL
		if previous != nil && req != nil && req.URL != nil {
			if strings.EqualFold(previous.Scheme, "https") && strings.EqualFold(req.URL.Scheme, "http") {
				return fmt.Errorf("redirect blocked: HTTPS downgrade from %q to %q", previous.String(), req.URL.String())
			}
			if !isAllowedRedirectHost(baseHost, req.URL.Hostname()) {
				return fmt.Errorf("redirect blocked: host %q is not allowed for base host %q", req.URL.Hostname(), baseHost)
			}
		}
		return nil
	}
}

func isAllowedRedirectHost(baseHost string, candidateHost string) bool {
	candidate := strings.ToLower(strings.TrimSpace(candidateHost))
	base := strings.ToLower(strings.TrimSpace(baseHost))
	if candidate == "" || base == "" {
		return false
	}
	if base == "huggingface.co" {
		if candidate == "huggingface.co" || strings.HasSuffix(candidate, ".huggingface.co") {
			return true
		}
		if candidate == "hf.co" || strings.HasSuffix(candidate, ".hf.co") {
			return true
		}
		return false
	}
	return candidate == base || strings.HasSuffix(candidate, "."+base)
}

func ensureModelAssets(cfg bootstrapConfig) (ModelAssets, error) {
	if strings.TrimSpace(cfg.repoID) == "" {
		return ModelAssets{}, fmt.Errorf("bootstrap repo ID cannot be empty")
	}
	if strings.TrimSpace(cfg.revision) == "" {
		return ModelAssets{}, fmt.Errorf("bootstrap revision cannot be empty")
	}
	if strings.TrimSpace(cfg.cacheDir) == "" {
		return ModelAssets{}, fmt.Errorf("bootstrap cache directory cannot be empty")
	}
	if strings.TrimSpace(cfg.baseURL) == "" {
		return ModelAssets{}, fmt.Errorf("bootstrap base URL cannot be empty")
	}
	if cfg.maxDownloadBytes <= 0 {
		return ModelAssets{}, fmt.Errorf("max download bytes must be > 0, got %d", cfg.maxDownloadBytes)
	}
	if cfg.httpClient == nil {
		cfg.httpClient = &http.Client{
			Timeout:       defaultHTTPTimeout,
			CheckRedirect: makeRedirectPolicy(cfg.baseURL),
		}
	}

	repoSlug := strings.ReplaceAll(cfg.repoID, "/", "--")
	revisionSlug := strings.ReplaceAll(cfg.revision, "/", "--")
	baseDir := filepath.Join(cfg.cacheDir, repoSlug, revisionSlug)
	if err := os.MkdirAll(baseDir, 0o700); err != nil {
		return ModelAssets{}, fmt.Errorf("failed to create bootstrap cache directory %q: %w", baseDir, err)
	}

	paths := map[string]string{}
	for _, spec := range bootstrapAssetSpecs {
		targetPath := filepath.Join(baseDir, spec.fileName)
		expectedChecksum := strings.ToLower(strings.TrimSpace(cfg.shaByFile[spec.fileName]))
		expectedSize := cfg.expectedSizeByFile[spec.fileName]

		if expectedSize < 0 {
			return ModelAssets{}, fmt.Errorf("invalid expected size for %s: %d", spec.fileName, expectedSize)
		}
		if err := ensureAssetFile(cfg, targetPath, spec.fileName, expectedChecksum, expectedSize); err != nil {
			return ModelAssets{}, err
		}
		paths[spec.fileName] = targetPath
	}

	return ModelAssets{
		TextModelPath:          paths[textModelFileName],
		VisionModelPath:        paths[visionModelFileName],
		TokenizerPath:          paths[tokenizerFileName],
		PreprocessorConfigPath: paths[preprocessorFileName],
	}, nil
}

func ensureAssetFile(cfg bootstrapConfig, destinationPath string, fileName string, expectedSHA256 string, expectedSize int64) (retErr error) {
	lockPath := destinationPath + ".lock"
	lock, err := acquireFileLock(lockPath, defaultLockWaitTimeout, defaultLockStaleAfter)
	if err != nil {
		return fmt.Errorf("failed to acquire bootstrap lock for %s: %w", fileName, err)
	}
	defer func() {
		if releaseErr := lock.Release(); releaseErr != nil && retErr == nil {
			retErr = fmt.Errorf("failed to release bootstrap lock for %s: %w", fileName, releaseErr)
		}
	}()

	if info, statErr := os.Stat(destinationPath); statErr == nil {
		switch {
		case expectedSize > 0 && info.Size() != expectedSize:
			if removeErr := os.Remove(destinationPath); removeErr != nil {
				return fmt.Errorf("failed to remove stale %s after size mismatch: %w", fileName, removeErr)
			}
		case cfg.verifySHA && expectedSHA256 != "":
			if verifyErr := verifyFileSHA256(destinationPath, expectedSHA256); verifyErr == nil {
				return nil
			}
			if removeErr := os.Remove(destinationPath); removeErr != nil {
				return fmt.Errorf("failed to remove stale %s after checksum mismatch: %w", fileName, removeErr)
			}
		default:
			return nil
		}
	}

	maxBytes := cfg.maxDownloadBytes
	if expectedSize > 0 && expectedSize < maxBytes {
		maxBytes = expectedSize
	}
	if maxBytes <= 0 {
		return fmt.Errorf("invalid max download limit for %s: %d", fileName, maxBytes)
	}

	assetURL := fmt.Sprintf(
		"%s/%s/resolve/%s/%s",
		strings.TrimRight(cfg.baseURL, "/"),
		cfg.repoID,
		cfg.revision,
		fileName,
	)

	if err := downloadFileWithRetry(cfg.httpClient, assetURL, destinationPath, cfg.hfToken, maxBytes, expectedSize); err != nil {
		return fmt.Errorf("failed to download %s: %w", fileName, err)
	}
	if cfg.verifySHA && expectedSHA256 != "" {
		if err := verifyFileSHA256(destinationPath, expectedSHA256); err != nil {
			_ = os.Remove(destinationPath)
			return fmt.Errorf("downloaded %s failed checksum verification: %w", fileName, err)
		}
	}
	return nil
}

func downloadFileWithRetry(client *http.Client, assetURL string, destinationPath string, hfToken string, maxBytes int64, expectedSize int64) error {
	var lastErr error
	const maxAttempts = 3
	for attempt := 1; attempt <= maxAttempts; attempt++ {
		if err := downloadFileOnce(client, assetURL, destinationPath, hfToken, maxBytes, expectedSize); err != nil {
			lastErr = err
			if !isRetryableDownloadError(err) || attempt == maxAttempts {
				break
			}
			time.Sleep(time.Duration(attempt) * 500 * time.Millisecond)
			continue
		}
		return nil
	}
	return fmt.Errorf("failed to download %s after %d attempts: %w", assetURL, maxAttempts, lastErr)
}

func downloadFileOnce(client *http.Client, assetURL string, destinationPath string, hfToken string, maxBytes int64, expectedSize int64) (err error) {
	if maxBytes <= 0 {
		return fmt.Errorf("max download bytes must be > 0, got %d", maxBytes)
	}

	req, err := http.NewRequest(http.MethodGet, assetURL, nil)
	if err != nil {
		return err
	}
	if strings.TrimSpace(hfToken) != "" {
		req.Header.Set("Authorization", "Bearer "+strings.TrimSpace(hfToken))
	}

	resp, err := client.Do(req)
	if err != nil {
		return err
	}
	defer func() {
		_ = resp.Body.Close()
	}()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(io.LimitReader(resp.Body, 2048))
		return &downloadStatusError{
			StatusCode: resp.StatusCode,
			URL:        assetURL,
			Body:       strings.TrimSpace(string(body)),
		}
	}
	if resp.ContentLength > 0 && resp.ContentLength > maxBytes {
		return fmt.Errorf("response content-length %d exceeds max allowed %d bytes for %s", resp.ContentLength, maxBytes, assetURL)
	}

	tempPath := destinationPath + ".tmp"
	if err := os.MkdirAll(filepath.Dir(destinationPath), 0o700); err != nil {
		return fmt.Errorf("failed to create destination directory: %w", err)
	}
	// #nosec G304 -- destination path is controlled by ensureAssetFile under the cache root.
	file, err := os.OpenFile(tempPath, os.O_CREATE|os.O_WRONLY|os.O_TRUNC, 0o600)
	if err != nil {
		return fmt.Errorf("failed to create temp file: %w", err)
	}
	defer func() {
		_ = file.Close()
		if err != nil {
			_ = os.Remove(tempPath)
		}
	}()

	limitedReader := io.LimitReader(resp.Body, maxBytes+1)
	written, copyErr := io.Copy(file, limitedReader)
	if copyErr != nil {
		return fmt.Errorf("failed to write response body: %w", copyErr)
	}
	if written > maxBytes {
		return fmt.Errorf("download exceeded max allowed size (%d > %d) for %s", written, maxBytes, assetURL)
	}
	if expectedSize > 0 && written != expectedSize {
		return fmt.Errorf("downloaded size mismatch for %s: got %d bytes, want %d bytes", assetURL, written, expectedSize)
	}

	if err = file.Sync(); err != nil {
		return fmt.Errorf("failed to flush temp file: %w", err)
	}
	if err = file.Close(); err != nil {
		return fmt.Errorf("failed to close temp file: %w", err)
	}
	if err = os.Rename(tempPath, destinationPath); err != nil {
		return fmt.Errorf("failed to move temp file into place: %w", err)
	}
	return nil
}

func isRetryableDownloadError(err error) bool {
	statusErr := (*downloadStatusError)(nil)
	if !errors.As(err, &statusErr) {
		return false
	}
	if statusErr.StatusCode == http.StatusRequestTimeout || statusErr.StatusCode == http.StatusTooManyRequests {
		return true
	}
	return statusErr.StatusCode >= 500 && statusErr.StatusCode <= 599
}

func acquireFileLock(lockPath string, waitTimeout time.Duration, staleAfter time.Duration) (*fileLock, error) {
	if strings.TrimSpace(lockPath) == "" {
		return nil, fmt.Errorf("lock path cannot be empty")
	}
	deadline := time.Now().Add(waitTimeout)
	for {
		// #nosec G304 -- lock path is derived from destination cache path in ensureAssetFile.
		lockFile, err := os.OpenFile(lockPath, os.O_CREATE|os.O_EXCL|os.O_WRONLY, 0o600)
		if err == nil {
			_, _ = io.WriteString(lockFile, fmt.Sprintf("pid=%d\n", os.Getpid()))
			_ = lockFile.Close()
			return &fileLock{path: lockPath}, nil
		}
		if !errors.Is(err, os.ErrExist) {
			return nil, err
		}

		// Best-effort stale lock cleanup for crashed writers.
		if staleAfter > 0 {
			if info, statErr := os.Stat(lockPath); statErr == nil {
				if time.Since(info.ModTime()) > staleAfter {
					_ = os.Remove(lockPath)
					continue
				}
			}
		}

		if time.Now().After(deadline) {
			return nil, fmt.Errorf("timed out waiting for lock %q", lockPath)
		}
		time.Sleep(200 * time.Millisecond)
	}
}

func verifyFileSHA256(path string, expectedSHA256 string) error {
	expected, err := parseSHA256(expectedSHA256)
	if err != nil {
		return err
	}
	// #nosec G304 -- path is controlled by ensureAssetFile under the cache root.
	file, err := os.Open(path)
	if err != nil {
		return fmt.Errorf("failed to open file %q: %w", path, err)
	}
	defer func() {
		_ = file.Close()
	}()

	hasher := sha256.New()
	if _, err := io.Copy(hasher, file); err != nil {
		return fmt.Errorf("failed to hash file %q: %w", path, err)
	}
	actual := hasher.Sum(nil)
	if !equalBytes(expected, actual) {
		return fmt.Errorf("checksum mismatch: got %s want %s", hex.EncodeToString(actual), strings.ToLower(expectedSHA256))
	}
	return nil
}

func validateSHA256(value string) error {
	_, err := parseSHA256(value)
	return err
}

func parseSHA256(value string) ([]byte, error) {
	trimmed := strings.TrimSpace(value)
	if len(trimmed) != 64 {
		return nil, fmt.Errorf("checksum must be 64 hex chars, got %d", len(trimmed))
	}
	decoded, err := hex.DecodeString(trimmed)
	if err != nil {
		return nil, fmt.Errorf("checksum is not valid hex: %w", err)
	}
	return decoded, nil
}

func validateAssetFileName(fileName string) error {
	switch strings.TrimSpace(fileName) {
	case textModelFileName, visionModelFileName, tokenizerFileName, preprocessorFileName:
		return nil
	default:
		return fmt.Errorf("unsupported bootstrap asset file %q", fileName)
	}
}

func equalBytes(a []byte, b []byte) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}
