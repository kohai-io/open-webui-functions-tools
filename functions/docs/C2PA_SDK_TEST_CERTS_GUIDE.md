# Using C2PA SDK Test Certificates for nano_banana_pro.py

## Important: Self-Signed Certificates Don't Work!

The C2PA SDK **does not allow self-signed certificates** for signing manifests. You must use either:
1. **SDK Test Certificates** (for development/testing)
2. **CA-Signed Certificates** (for production)

## Option 1: SDK Test Certificates (Recommended for Testing)

The C2PA Python SDK provides official test certificates that work with the library.

### Download Test Certificates

```bash
cd c:\Users\Robert\dev\AI\open-webui\open-webui

# Create certs directory
mkdir certs
cd certs

# Download test certificates from c2pa-python repository
# ES256 (Elliptic Curve) - Recommended
curl -O https://raw.githubusercontent.com/contentauth/c2pa-python/main/tests/fixtures/es256_certs.pem
curl -O https://raw.githubusercontent.com/contentauth/c2pa-python/main/tests/fixtures/es256_private.key

# Alternative: PS256 (RSA)
# curl -O https://raw.githubusercontent.com/contentauth/c2pa-python/main/tests/fixtures/ps256.pem
# curl -O https://raw.githubusercontent.com/contentauth/c2pa-python/main/tests/fixtures/ps256.key
```

### Configure in Open WebUI

#### Option A: Using File Paths

1. Open Open WebUI → **Workspace → Functions**
2. Find **"Gemini 3 Pro Image"** pipeline
3. Configure these valves:

| Valve | Value |
|-------|-------|
| **ENABLE_C2PA_SIGNING** | `true` |
| **C2PA_CERT_PATH** | `C:\Users\Robert\dev\AI\open-webui\open-webui\certs\es256_certs.pem` |
| **C2PA_KEY_PATH** | `C:\Users\Robert\dev\AI\open-webui\open-webui\certs\es256_private.key` |
| **C2PA_TSA_URL** | *(leave empty for testing)* |
| **C2PA_TRAINING_POLICY** | `notAllowed` |

#### Option B: Using Content (More Secure - Recommended)

```bash
# Read certificate
cat es256_certs.pem
# Copy the entire output

# Read private key
cat es256_private.key
# Copy the entire output
```

Then in Open WebUI:

| Valve | Value |
|-------|-------|
| **ENABLE_C2PA_SIGNING** | `true` |
| **C2PA_CERT_CONTENT** | *Paste certificate content* |
| **C2PA_KEY_CONTENT** | *Paste private key content* |
| **C2PA_TSA_URL** | *(leave empty for testing)* |
| **C2PA_TRAINING_POLICY** | `notAllowed` |

### Test It

Generate an image and check logs:

```
[DEBUG] Signing image with C2PA (size: 2027398 bytes)
[DEBUG] Using C2PA certificate from content (encrypted)
[DEBUG] Using C2PA private key from content (encrypted)
[DEBUG] C2PA signing successful! Manifest size: 12345 bytes
[DEBUG] Signed image size: 2039743 bytes (original: 2027398 bytes)
```

### Verify the Manifest

```bash
# Download the generated image
# Then verify with c2pa-tool
c2pa-tool signed_image.png
```

**Expected output:**
```
✅ C2PA Manifest found!
⚠️  Warning: "The Content Credential issuer couldn't be recognized."

Title: AI Generated Image - 2025-12-05T17:08:00Z
Claim Generator: OpenWebUI/NanoBananaPro
Generator: gemini-3-pro-image-preview
```

The warning is **normal for test certificates** - they're not from a recognized Certificate Authority.

## Option 2: Production Certificates

For production deployments, obtain certificates from:

### Adobe Content Authenticity Initiative
- Apply at: https://contentauthenticity.org/
- Free for qualifying organizations
- Industry-standard trusted certificates

### Commercial Certificate Authorities
- **DigiCert**: https://www.digicert.com/
- **GlobalSign**: https://www.globalsign.com/
- **Sectigo**: https://sectigo.com/

### Requirements for Production Certificates

1. **X.509 Certificate** compatible with C2PA specification
2. **Signing algorithm**: ES256 (ECDSA) or PS256 (RSA-PSS) recommended
3. **Certificate chain**: Include intermediate certificates if needed
4. **Private key**: PKCS#8 format (PEM encoded with `-----BEGIN PRIVATE KEY-----`)

## Available Test Certificate Types

The c2pa-python repository provides test certificates for multiple algorithms:

| File | Algorithm | Use Case |
|------|-----------|----------|
| `es256_certs.pem` | ES256 (ECDSA P-256) | **Recommended** - Fast, small signatures |
| `es256_private.key` | ES256 Private Key | Pairs with es256_certs.pem |
| `ps256.pem` | PS256 (RSA-PSS 2048) | Alternative option |
| `ps256.key` | PS256 Private Key | Pairs with ps256.pem |
| `ed25519.pem` | Ed25519 | Modern elliptic curve |
| `ed25519.key` | Ed25519 Private Key | Pairs with ed25519.pem |

### Choosing an Algorithm

- **ES256**: Best choice for most use cases (fast, secure, small)
- **PS256**: Use if you need RSA compatibility
- **Ed25519**: Modern alternative, not all systems support it yet

## Key Format Requirements

⚠️ **Critical**: Private keys MUST be in **PKCS#8 format**

The SDK test keys are already in the correct format. If you generate your own keys:

```bash
# Wrong format (won't work)
openssl ecparam -name prime256v1 -genkey -out key.pem
# Creates: -----BEGIN EC PRIVATE KEY-----

# Correct format (will work)
openssl genpkey -algorithm EC -pkeyopt ec_paramgen_curve:P-256 -out key.pem
# Creates: -----BEGIN PRIVATE KEY-----
```

## Timestamp Authority (Optional)

For production, add a TSA URL:

```
C2PA_TSA_URL: http://timestamp.digicert.com
```

This adds a trusted timestamp proving when the image was signed.

## Security Notes

### Test Certificates
- ✅ Perfect for development and testing
- ✅ Work with C2PA SDK without additional configuration
- ⚠️  Display "issuer couldn't be recognized" warning
- ❌ **Do NOT use in production** - not trusted by verifiers

### Production Certificates
- ✅ Trusted by Content Credentials Verify tool
- ✅ Recognized by Adobe products and industry tools
- ✅ No warning messages for end users
- ✅ Legal non-repudiation

## Troubleshooting

### "Signature: the certificate was self-signed"

**Cause**: You're using an OpenSSL-generated self-signed certificate

**Solution**: Download and use the SDK test certificates instead (see above)

### "Invalid ES256 private key: expecting 'PRIVATE KEY'"

**Cause**: Wrong key format (EC PRIVATE KEY vs PRIVATE KEY)

**Solution**: Use SDK test certificates or convert:
```bash
openssl pkcs8 -topk8 -nocrypt -in old_key.pem -out new_key.pem
```

### "C2PA signing successful" but Adobe Verify shows "No credentials found"

**Possible causes**:
- Image was re-encoded (JPEG quality change strips C2PA data)
- Image was screenshot or saved differently
- C2PA data was accidentally stripped

**Solution**: Download the image directly from Open WebUI without re-saving

## Next Steps

1. **Download SDK test certificates** (es256_certs.pem, es256_private.key)
2. **Paste into Open WebUI** pipeline settings (C2PA_CERT_CONTENT, C2PA_KEY_CONTENT)
3. **Enable C2PA signing** (ENABLE_C2PA_SIGNING: true)
4. **Enable debug mode** to see signing activity
5. **Generate an image** and verify it works
6. **For production**: Obtain proper CA-signed certificates

## References

- **SDK Test Certificates**: https://github.com/contentauth/c2pa-python/tree/main/tests/fixtures
- **C2PA Specification**: https://c2pa.org/specifications/
- **Content Credentials**: https://contentcredentials.org/
- **Adobe CAI**: https://contentauthenticity.org/
