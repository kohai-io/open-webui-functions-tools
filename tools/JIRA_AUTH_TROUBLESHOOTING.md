# Jira Tool Authentication Troubleshooting

## ✅ RESOLVED: PAT Authentication Method

**Error:** "Failed to parse Connect Session Auth Token"  
**Cause:** Jira Cloud PATs must use Basic Auth (email:PAT), NOT Bearer token  
**Fix:** Tool now uses Basic Auth for PAT authentication with email as username

### Correct Configuration for PAT:
- **Username**: Your Jira email address (e.g., user@company.com)
- **Password**: Leave empty
- **PAT**: Your Personal Access Token

The tool has been updated (v1.1.1). Please reload/restart Open WebUI.

### Update (v1.1.1): Search Endpoint Migration
Jira Cloud deprecated the `/rest/api/3/search` endpoint in favor of `/rest/api/3/search/jql`. The tool now uses the correct endpoint for all search operations.

### Update (v1.1.0): API v2 Deprecation
Jira Cloud deprecated the `/rest/api/2/search` endpoint in 2025. The tool now uses API v3 for **all authentication methods** (both PAT and Basic Auth). This ensures compatibility with current Jira Cloud requirements.

---

## Issue: 403 Permission Error with PAT

### Enhanced Logging Added

The tool now includes enhanced logging to diagnose authentication issues:

1. **PAT length verification** - Shows if PAT is being decrypted properly
2. **Authentication method** - Shows whether using PAT or Basic Auth
3. **Response details** - Shows response headers and body for 403 errors

### Common Causes & Solutions

#### 1. **Jira Data Center vs Jira Cloud**

Your instance (`itvplc.jira.com`) appears to be **Jira Data Center/Server**, not Jira Cloud.

**Important:** Jira Data Center may require different authentication:
- ✅ Jira Cloud: `Bearer {token}` header
- ❌ Jira Data Center: May need Basic Auth with PAT as password

**Solution:** Try using Basic Auth instead:
- **Username**: Your Jira username/email
- **Password**: Leave empty
- **PAT**: Enter your PAT here

If that doesn't work, try:
- **Username**: Your Jira username/email
- **Password**: Your PAT token
- **PAT**: Leave empty

#### 2. **PAT Encryption Issue**

If the PAT length shows as 0 in logs, the encryption/decryption failed.

**Check:**
```bash
# Verify WEBUI_SECRET_KEY is set
echo $WEBUI_SECRET_KEY
```

**Solution:**
- Ensure `WEBUI_SECRET_KEY` environment variable is set
- Re-enter your PAT in the tool settings
- Restart Open WebUI to reload environment variables

#### 3. **PAT Permissions**

Your PAT may not have the required permissions.

**Required Jira Permissions:**
- `READ` - Browse projects and issues
- `WRITE` - Create and edit issues (if needed)
- `Browse Projects` permission on the specific project

**Solution:**
- Check PAT permissions in Jira settings
- Verify you have access to project `AAHM`
- Try with a different PAT with broader permissions

#### 4. **PAT Format**

**Jira Cloud PAT format:**
- Typically starts with `ATATT` or `ATCTT`
- Usually 204+ characters long

**Jira Data Center PAT format:**
- May vary depending on version
- Check your Jira version and documentation

### Testing Steps

1. **Check the new logs:**
   ```
   # Look for these lines in your logs:
   - "PAT provided: True/False, PAT length: X"
   - "Using PAT authentication (token length: X)"
   - "403 Error - Response headers: {...}"
   - "403 Error - Response body: ..."
   ```

2. **Verify token in Postman/curl:**
   ```bash
   # Test with Bearer token (Jira Cloud)
   curl -H "Authorization: Bearer YOUR_PAT" \
        https://itvplc.jira.com/rest/api/latest/issue/AAHM-140
   
   # Test with Basic Auth (Jira Data Center)
   curl -u "username:YOUR_PAT" \
        https://itvplc.jira.com/rest/api/latest/issue/AAHM-140
   ```

3. **Check Jira version:**
   - Navigate to: `https://itvplc.jira.com/rest/api/2/serverInfo`
   - Look for `"deploymentType": "Cloud"` or `"Server"`

### Next Steps

1. **Restart Open WebUI** to pick up the updated tool
2. **Check logs** for the new debug output
3. **Try the test curl commands** above to verify authentication works
4. **Report back** with:
   - PAT length from logs
   - Authentication method being used
   - Response headers/body from 403 error
   - Results from curl tests

### Alternative: Use API Token (Jira Cloud only)

If using Jira Cloud, you can also use an API token:
- Generate at: https://id.atlassian.com/manage/api-tokens
- Use with Basic Auth: username + API token as password
- In tool settings:
  - **Username**: Your Atlassian email
  - **Password**: Your API token
  - **PAT**: Leave empty
