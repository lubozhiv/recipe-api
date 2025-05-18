import os
import dotenv
from typing import Dict
from github import Github
from llama_index.llms.openai import OpenAI
from llama_index.core.tools import FunctionTool
from llama_index.core.agent.workflow import AgentWorkflow, FunctionAgent
from llama_index.core.workflow import Context
import asyncio
from llama_index.core.prompts import RichPromptTemplate
from llama_index.core.agent.workflow import AgentOutput, ToolCall, ToolCallResult


def debug_print(title, content):
    """Helper function for consistent debug output"""
    print(f"\n=== DEBUG: {title} ===")
    print(content)
    print("=" * (len(title) + 12) + "\n")


# Debug environment setup
debug_print("Environment Setup", "Loading environment variables and initializing components")

# Load environment variables
dotenv.load_dotenv()

# Debug environment variables
env_vars_to_check = [
    "GITHUB_TOKEN",
    "OPENAI_MODEL",
    "OPENAI_API_KEY",
    "OPENAI_BASE_URL",
    "REPOSITORY",
    "PR_NUMBER"
]

debug_print("Environment Variables", {var: os.getenv(var) for var in env_vars_to_check})

# Initialize GitHub client
# try:
#     GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
#     debug_print("GitHub Token", "Present" if GITHUB_TOKEN else "Missing")
#
#     git = Github(GITHUB_TOKEN) if GITHUB_TOKEN else None
#     if git:
#         debug_print("GitHub Client", "Initialized successfully")
#         debug_print(f"GitHub token {GITHUB_TOKEN}")
#         # Test GitHub connection
#         try:
#             user = git.get_user()
#             debug_print("GitHub Connection Test", f"Connected as: {user.login}")
#         except Exception as e:
#             debug_print("GitHub Connection Error", str(e))
#     else:
#         debug_print("GitHub Client", "Not initialized - missing token")
# except Exception as e:
#     debug_print("GitHub Initialization Error", str(e))
#     git = None

# Initialize LLM
model = os.getenv("OPENAI_MODEL") or "gpt-4o-mini"
api_key = os.getenv("OPENAI_API_KEY") or "sk-vetvThlLdl7Zvs4sEd-e2Q"
if not api_key:
    raise ValueError("OPENAI_API_KEY environment variable must be set")
api_base = os.getenv("OPENAI_BASE_URL") or "https://litellm.aks-hs-prod.int.hyperskill.org"

try:
    llm = OpenAI(
        model=model,
        api_key=api_key,
        api_base=api_base,
    )
    debug_print("LLM Initialization", "Success")
    debug_print("LLM Config", {
        "model": os.getenv("OPENAI_MODEL"),
        "api_base": os.getenv("OPENAI_BASE_URL"),
        "api_key": "Present" if os.getenv("OPENAI_API_KEY") else "Missing"
    })
except Exception as e:
    debug_print("LLM Initialization Error", str(e))
    llm = None

# Get repository information
full_repo_name = os.getenv("REPOSITORY")  # Format: "username/repo-name"
debug_print("Repository Info", f"Full repo name: {full_repo_name}")

if git is not None:
    try:
        repo = git.get_repo(full_repo_name)
        debug_print("Repository Access", f"Successfully accessed: {repo.full_name}")
    except Exception as e:
        debug_print("Repository Access Error", str(e))
        repo = None
else:
    repo = None
    debug_print("Repository Access", "Skipped - GitHub client not initialized")


# State management tools
async def add_context_to_state(ctx: Context, context: str) -> Dict:
    debug_print("add_context_to_state", f"Adding context: {context[:100]}...")
    try:
        state = await ctx.get("state")
        state["gathered_contexts"] = context
        await ctx.set("state", state)
        debug_print("State Update", "Context added successfully")
        return {"success": True, "message": "Context added to state"}
    except Exception as e:
        debug_print("State Update Error", str(e))
        return {"success": False, "error": str(e)}


add_context_to_state_tool = FunctionTool.from_defaults(
    fn=add_context_to_state,
    name="add_context_to_state",
    description="Adds the gathered context to the workflow state."
)


async def add_comment_to_state(ctx: Context, draft_comment: str) -> Dict:
    debug_print("add_comment_to_state", f"Adding draft comment: {draft_comment[:100]}...")
    try:
        state = await ctx.get("state")
        state["draft_comment"] = draft_comment
        await ctx.set("state", state)
        debug_print("State Update", "Draft comment added successfully")
        return {"success": True, "message": "Draft comment added to state"}
    except Exception as e:
        debug_print("State Update Error", str(e))
        return {"success": False, "error": str(e)}


add_comment_to_state_tool = FunctionTool.from_defaults(
    fn=add_comment_to_state,
    name="add_comment_to_state",
    description="Adds the draft PR comment to the workflow state."
)


# GitHub interaction tools with enhanced debugging
async def get_pr_details(pr_number: int) -> Dict:
    debug_print("get_pr_details", f"Fetching details for PR #{pr_number}")

    if not git or not repo:
        error_msg = "GitHub client not initialized"
        debug_print("Error", error_msg)
        return {"error": error_msg}

    try:
        pull_request = repo.get_pull(pr_number)
        commit_SHAs = [c.sha for c in pull_request.get_commits()]
        pr_body = pull_request.body if pull_request.body else ""

        result = {
            "author": pull_request.user.login,
            "title": pull_request.title,
            "body": pr_body,
            "diff_url": pull_request.diff_url,
            "state": pull_request.state,
            "head_sha": pull_request.head.sha,
            "commit_SHAs": commit_SHAs,
            "base_ref": pull_request.base.ref,
            "head_ref": pull_request.head.ref,
        }

        debug_print("PR Details Result", {
            "title": result['title'],
            "author": result['author'],
            "body_length": len(result['body']),
            "num_commits": len(result['commit_SHAs'])
        })

        return result
    except Exception as e:
        debug_print("PR Details Error", str(e))
        return {"error": str(e)}


pr_details_tool = FunctionTool.from_defaults(
    fn=get_pr_details,
    name="get_pr_details",
    description="Fetches details about a pull request."
)


async def get_file_contents(file_path: str) -> Dict:
    debug_print("get_file_contents", f"Fetching contents for file: {file_path}")

    if not git or not repo:
        error_msg = "GitHub client not initialized"
        debug_print("Error", error_msg)
        return {"error": error_msg}

    try:
        file_content = repo.get_contents(file_path)
        result = {
            "content": file_content.decoded_content.decode('utf-8'),
            "path": file_content.path,
            "sha": file_content.sha,
            "size": file_content.size,
        }

        debug_print("File Contents Result", {
            "path": result['path'],
            "size": result['size'],
            "content_sample": result['content'][:100] + "..." if result['content'] else "Empty"
        })

        return result
    except Exception as e:
        debug_print("File Contents Error", str(e))
        return {"error": str(e)}


file_contents_tool = FunctionTool.from_defaults(
    fn=get_file_contents,
    name="get_file_contents",
    description="Fetches the contents of a file from the repository."
)


async def get_commit_details(commit_sha: str) -> Dict:
    debug_print("get_commit_details", f"Fetching details for commit: {commit_sha}")

    if not git or not repo:
        error_msg = "GitHub client not initialized"
        debug_print("Error", error_msg)
        return {"error": error_msg}

    try:
        commit = repo.get_commit(commit_sha)
        changed_files = []
        for f in commit.files:
            changed_file = {
                "filename": f.filename,
                "status": f.status,
                "additions": f.additions,
                "deletions": f.deletions,
                "changes": f.changes,
                "patch": f.patch if hasattr(f, 'patch') else None,
            }
            changed_files.append(changed_file)

        result = {
            "sha": commit.sha,
            "author": commit.author.login if commit.author else None,
            "message": commit.commit.message,
            "changed_files": changed_files,
        }

        debug_print("Commit Details Result", {
            "sha": result['sha'],
            "author": result['author'],
            "message": result['message'],
            "num_changed_files": len(result['changed_files'])
        })

        return result
    except Exception as e:
        debug_print("Commit Details Error", str(e))
        return {"error": str(e)}


commit_details_tool = FunctionTool.from_defaults(
    fn=get_commit_details,
    name="get_commit_details",
    description="Fetches details about a commit including changed files."
)


async def add_final_review_to_state(ctx: Context, final_review: str) -> Dict:
    debug_print("add_final_review_to_state", f"Adding final review: {final_review[:100]}...")
    try:
        state = await ctx.get("state")
        state["final_review_comment"] = final_review
        await ctx.set("state", state)
        debug_print("State Update", "Final review added successfully")
        return {"success": True, "message": "Final review added to state"}
    except Exception as e:
        debug_print("State Update Error", str(e))
        return {"success": False, "error": str(e)}


add_final_review_to_state_tool = FunctionTool.from_defaults(
    fn=add_final_review_to_state,
    name="add_final_review_to_state",
    description="Adds the final approved review comment to the workflow state."
)


async def post_review_to_github(pr_number: int, body: str) -> Dict:
    debug_print("post_review_to_github", f"Posting review to PR #{pr_number}")

    if not git or not repo:
        error_msg = "GitHub client not initialized"
        debug_print("Error", error_msg)
        return {"error": error_msg}

    try:
        pull_request = repo.get_pull(pr_number)
        debug_print("Review Content", f"Body length: {len(body)} characters")
        debug_print("Review Preview", body[:200] + "..." if len(body) > 200 else body)

        review = pull_request.create_review(body=body)

        debug_print("Review Posted", {
            "success": True,
            "review_id": review.id,
            "html_url": review.html_url
        })

        return {
            "success": True,
            "message": "Review posted successfully",
            "review_id": review.id,
            "html_url": review.html_url
        }
    except Exception as e:
        debug_print("Post Review Error", str(e))
        return {"error": str(e)}


post_review_tool = FunctionTool.from_defaults(
    fn=post_review_to_github,
    name="post_review_to_github",
    description="Posts a review comment to the specified GitHub pull request."
)

# Agents with debug info
context_agent_system_prompt = """You are the context gathering agent. Your responsibilities:
1. Gather PR details (author, title, body, diff_url, state, head_sha)
2. Collect changed files information
3. Retrieve any requested file contents
Once you have the requested info, hand control back to the requesting agent."""

context_agent = FunctionAgent(
    llm=llm,
    name="ContextAgent",
    description="Gathers context from GitHub for PR review.",
    tools=[pr_details_tool, file_contents_tool, commit_details_tool, add_context_to_state_tool],
    system_prompt=context_agent_system_prompt,
    can_handoff_to=["CommentorAgent", "ReviewAndPostingAgent"]
)

commentor_agent_system_prompt = """You are the commentor agent that writes PR review comments. You must:
1. FIRST request all needed information from ContextAgent using get_pr_details and other tools
2. THEN create a comprehensive draft review
3. FINALLY use add_comment_to_state tool to store the draft
4. MUST hand off to ReviewAndPostingAgent with message: "Draft review ready for approval\""""

commentor_agent = FunctionAgent(
    llm=llm,
    name="CommentorAgent",
    description="Drafts PR review comments.",
    tools=[add_comment_to_state_tool],
    system_prompt=commentor_agent_system_prompt,
    can_handoff_to=["ContextAgent", "ReviewAndPostingAgent"]
)

review_and_posting_agent_system_prompt = """You are the Review and Posting agent. You must:
1. Acknowledge receiving control: 
   "I am now handling the request as ReviewAndPostingAgent."
2. If 'final_review_comment' is not set in state, request a review from CommentorAgent.
3. Verify the draft review meets all criteria:
   - Proper length and format
   - Covers all required aspects
   - Contains specific improvement suggestions
4. Request rewrites from CommentorAgent if needed
5. Post to GitHub when satisfied
6. Use the post_review_to_github tool for final posting"""

review_and_posting_agent = FunctionAgent(
    llm=llm,
    name="ReviewAndPostingAgent",
    description="Reviews and posts final PR comments to GitHub.",
    tools=[add_final_review_to_state_tool, post_review_tool],
    system_prompt=review_and_posting_agent_system_prompt,
    can_handoff_to=["CommentorAgent"]
)

# Workflow setup
workflow_agent = AgentWorkflow(
    agents=[context_agent, commentor_agent, review_and_posting_agent],
    root_agent="ReviewAndPostingAgent",
    initial_state={
        "gathered_contexts": "",
        "draft_comment": "",
        "final_review_comment": ""
    },
)


async def main():
    debug_print("Workflow Start", "Initializing PR review workflow")

    # Get PR number from environment variable
    pr_number = os.getenv("PR_NUMBER")
    debug_print("PR Number", pr_number if pr_number else "PR_NUMBER not set")

    if not pr_number:
        debug_print("Error", "PR_NUMBER environment variable not set")
        return

    debug_print("Workflow Parameters", {
        "repository": full_repo_name,
        "pr_number": pr_number,
        "llm_initialized": llm is not None,
        "github_initialized": git is not None
    })

    ctx = Context(workflow_agent)
    query = f"Write a review for PR #{pr_number}"

    # Extract PR number
    import re
    pr_match = re.search(r'#?(\d+)', query)
    if pr_match:
        pr_number = int(pr_match.group(1))
        if repo:
            try:
                pull_request = repo.get_pull(pr_number)
                pr_body = pull_request.body if pull_request.body else ""
                debug_print("Direct PR Fetch", {
                    "title": pull_request.title,
                    "body_length": len(pr_body),
                    "state": pull_request.state
                })
            except Exception as e:
                debug_print("Direct PR Fetch Error", str(e))

    prompt = RichPromptTemplate(f"Write a review for PR #{pr_number}")
    handler = workflow_agent.run(prompt.format(), ctx=ctx)

    current_agent = None
    async for event in handler.stream_events():
        if hasattr(event, "current_agent_name") and event.current_agent_name != current_agent:
            current_agent = event.current_agent_name
            debug_print("Agent Change", f"New active agent: {current_agent}")

            # Explicit handoff logging
            debug_print("Handoff Reason",
                        "Initial request" if current_agent == "ReviewAndPostingAgent"
                        else "Requested information")

        elif isinstance(event, AgentOutput):
            if event.response and event.response.content:
                debug_print("Agent Output", event.response.content)
            if event.tool_calls:
                debug_print("Tool Selection", {
                    "tools": [call.tool_name for call in event.tool_calls],
                    "args": [call.tool_kwargs for call in event.tool_calls]
                })
                for call in event.tool_calls:
                    if call.tool_name == "handoff":
                        debug_print("Agent Handoff", {
                            "to_agent": call.tool_kwargs.get('to_agent'),
                            "reason": call.tool_kwargs.get('reason', 'Workflow progression')
                        })

        elif isinstance(event, ToolCallResult):
            debug_print("Tool Result", {
                "tool_name": event.tool_name,
                "output": event.tool_output if isinstance(event.tool_output, str)
                else str(event.tool_output)[:200] + "..."
                if len(str(event.tool_output)) > 200 else str(event.tool_output)
            })

        elif isinstance(event, ToolCall):
            debug_print("Tool Call", {
                "tool_name": event.tool_name,
                "arguments": event.tool_kwargs
            })

    state = await ctx.get("state")
    debug_print("Final State", state)
    debug_print("Workflow Completion", "Workflow finished")


if __name__ == "__main__":
    try:
        debug_print("Application Start", "Starting async main function")
        asyncio.run(main())
    except Exception as e:
        debug_print("Application Error", f"Critical failure: {str(e)}")
        # Print full traceback for critical errors
        import traceback

        debug_print("Stack Trace", traceback.format_exc())
    finally:
        if git:
            git.close()
            debug_print("GitHub Client", "Closed connection")
        debug_print("Application End", "Process completed")