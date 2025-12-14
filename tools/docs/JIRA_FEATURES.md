# Jira Agent for Open WebUI - Features Overview

## 🎯 Core Features (v1.0.9)

### ✅ **Implemented & Available**

#### 1. **View Issue Details** (`get_issue`)
Retrieve comprehensive information about any Jira issue:
- Issue summary, description, status, priority
- Assignee, reporter, and creation/update dates
- All comments with authors and timestamps
- Direct clickable links to issues
- Rendered HTML descriptions

**Usage Examples:**
- "Show me AAHM-140"
- "Get details for PROJ-123"
- "What's the status of DEMO-456?"

---

#### 2. **Search Issues** (`search_issues`)
Search for issues using JQL or free text:
- Supports full Jira Query Language (JQL)
- Free text search across issues
- Configurable result limits (default: 10)
- Returns formatted table with key info

**Usage Examples:**
- "Find all high priority bugs"
- "Search for issues assigned to me"
- "Show issues: status = 'In Progress' AND project = DEMO"
- "Find authentication issues"

**JQL Examples:**
```jql
# Issues assigned to current user
assignee = currentUser() ORDER BY updated DESC

# High priority issues created this week
priority = High AND created >= -7d

# In Progress issues in specific project
project = DEMO AND status = 'In Progress'

# Bugs updated recently
type = Bug AND updated >= -3d
```

---

#### 3. **Create Issues** (`create_issue`)
Create new Jira issues programmatically:
- Support for all issue types (Bug, Task, Story, etc.)
- Set priority, summary, and description
- Automatic issue key generation
- Returns link to newly created issue

**Usage Examples:**
- "Create a bug in DEMO about login timeout"
- "Create task in PROJ: Update documentation"
- "New story: As a user I want to reset my password"

**Parameters:**
- `project_key` - Project code (required)
- `summary` - Issue title (required)
- `description` - Detailed description (required)
- `issue_type` - Bug, Task, Story, etc. (default: Task)
- `priority` - High, Medium, Low (optional)

---

#### 4. **Add Comments** (`add_comment`)
Add comments to existing issues:
- Works with both legacy and ADF (Atlassian Document Format)
- Returns timestamp of comment creation
- Includes link to the issue
- Supports markdown formatting

**Usage Examples:**
- "Add comment to AAHM-140: Investigation complete"
- "Comment on DEMO-123: Fixed in version 2.1"
- "Add note to PROJ-456: Needs review"

---

#### 5. **Help Documentation** (`help`)
Built-in comprehensive help system:
- Lists all available features
- Provides usage examples
- Shows JQL search patterns
- Documents authentication setup
- Includes tips and best practices

**Usage:**
- "What can you do?"
- "Show Jira help"
- "List Jira features"

---

## 🔐 Security Features

### **Credential Encryption**
- ✅ **At-Rest Encryption**: All credentials encrypted in database
- ✅ **Fernet Encryption**: Military-grade symmetric encryption
- ✅ **Key Derivation**: Uses `WEBUI_SECRET_KEY` environment variable
- ✅ **Migration Support**: Seamlessly upgrades plain text to encrypted
- ✅ **Graceful Fallback**: Works without encryption key (with warning)

### **Authentication Methods**
- ✅ **API Tokens**: Recommended for Jira Cloud
- ✅ **Personal Access Tokens (PAT)**: Alternative auth method
- ✅ **Basic Auth**: Standard username/password
- ✅ **Auto-Detection**: Automatically uses correct auth method

### **Security Best Practices**
- 🔒 Credentials never logged in plain text
- 🔒 Stored in closures, not instance variables
- 🔒 Decrypted only when needed
- 🔒 Prefix-based encryption detection
- 🔒 TLS/HTTPS enforced for all API calls

---

## 📊 Data Presentation

### **Issue Details Format**
```
## [PROJ-123] Issue Title

✅ Done  🔥 High  📋 Bug  🕒 Oct 15, 2025 3:45 PM  
🔄 Oct 17, 2025 9:15 PM  🙋 John Doe  🕵️‍♂️ Jane Smith

[Rendered HTML description]

### 💬 Comments (3)
[All comments with authors and timestamps]
```

### **Search Results Format**
```
### Found 25 issues (showing 10)

| Key | Summary | Status | Type | Priority | Updated |
|-----|---------|--------|------|----------|---------|
| [PROJ-123](link) | Issue title | In Progress | Bug | High | Oct 17, 2025 |
```

---

## 🚀 API Coverage

### **Jira REST API v3**
- ✅ `/issue/{issueId}` - Get issue details
- ✅ `/search` - Search with JQL
- ✅ `/issue` - Create new issues
- ✅ `/issue/{issueId}/comment` - Get/add comments
- ⚙️ `/project` - List projects (implemented, not exposed)
- ⚙️ `/issuetype` - Get issue types (implemented, not exposed)
- ⚙️ `/priority` - Get priorities (implemented, not exposed)
- ⚙️ `/issue/{issueId}/assignee` - Assign issues (implemented, not exposed)
- ⚙️ `/issue/{issueId}/transitions` - Update status (implemented, not exposed)

**Legend:**
- ✅ Available to users
- ⚙️ Implemented but not exposed as tool function (coming soon)

---

## 🛠️ Underlying Capabilities (Not Yet Exposed)

These features are fully implemented in the code but not yet available as tool functions:

### **1. Project Management**
- Get list of available projects (`get_projects`)
- View project details and permissions

### **2. Issue Type Management**
- List all issue types (`get_issue_types`)
- Filter by project
- Get type-specific fields

### **3. Priority Management**
- List available priorities (`get_priorities`)
- Includes IDs and names

### **4. Assignment**
- Assign issues to users (`assign_issue`)
- Unassign issues
- Support for usernames and emails

### **5. Status Transitions**
- Update issue status (`update_issue_status`)
- Get available transitions (`get_available_transitions`)
- Support for workflow steps

### **6. Advanced Formatting**
- Emit formatted tables (`emit_table`)
- Source citations (`emit_source`)
- Progress indicators (`emit_status`)

---

## 💡 Tips & Best Practices

### **Searching**
1. **Use JQL for precision**: JQL queries are more powerful than free text
2. **Start broad, then narrow**: Search first, then get details
3. **Save common queries**: Document frequently used JQL patterns
4. **Use ORDER BY**: Sort results by relevance (updated, created, priority)

### **Creating Issues**
1. **Be descriptive**: Good summaries help with searches
2. **Use markdown**: Descriptions support markdown formatting
3. **Set priority early**: Helps with triage and visibility
4. **Link related issues**: Mention other issue keys in descriptions

### **Comments**
1. **Track progress**: Use comments for status updates
2. **Tag users**: Mention @username to notify team members
3. **Date stamp**: Tool automatically includes timestamps
4. **Link evidence**: Include URLs, code snippets, logs

### **Authentication**
1. **Use API Tokens**: More secure than passwords
2. **Rotate regularly**: Generate new tokens periodically
3. **Set WEBUI_SECRET_KEY**: Enable encryption
4. **Test with curl**: Verify auth before troubleshooting tool

---

## 📈 Performance & Limits

### **Rate Limiting**
- Jira Cloud: ~300 requests per minute (per user)
- Tool respects Jira's rate limits
- Automatic retry not yet implemented

### **Result Limits**
- Search: Default 10 results, max configurable
- Comments: Returns all comments (no pagination)
- Projects: Returns all accessible projects

### **Timeouts**
- Default: 30 seconds per request
- Configurable in code if needed

---

## 🔧 Configuration

### **Required Settings**
- **Username**: Your Jira email address
- **Password/PAT**: API Token or Personal Access Token
- **Base URL**: Your Jira instance URL (e.g., https://company.atlassian.net)

### **Optional Environment Variables**
- `WEBUI_SECRET_KEY`: For credential encryption (recommended)
- `LOG_LEVEL`: For debug logging

---

## 📝 Changelog Highlights

- **v1.0.9**: Added comprehensive help command
- **v1.0.8**: Enhanced debugging for troubleshooting
- **v1.0.7**: Fixed PAT authentication (Basic Auth)
- **v1.0.4**: Added credential encryption
- **v1.0.3**: Improved formatting and HTML handling
- **v1.0.0**: Initial release

---

## 🎓 Learning Resources

### **JQL (Jira Query Language)**
- [Official JQL Guide](https://support.atlassian.com/jira-service-management-cloud/docs/use-advanced-search-with-jira-query-language-jql/)
- [JQL Functions Reference](https://support.atlassian.com/jira-software-cloud/docs/jql-functions/)
- [JQL Keywords and Operators](https://support.atlassian.com/jira-software-cloud/docs/jql-keywords/)

### **Jira REST API**
- [API v3 Documentation](https://developer.atlassian.com/cloud/jira/platform/rest/v3/intro/)
- [Basic Auth Setup](https://developer.atlassian.com/cloud/jira/platform/basic-auth-for-rest-apis/)
- [API Tokens](https://support.atlassian.com/atlassian-account/docs/manage-api-tokens-for-your-atlassian-account/)

---

**Status**: ✅ Production Ready | **Version**: 1.0.9 | **License**: As per repository
