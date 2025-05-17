import os
import dotenv
from typing import Dict, List
from github import Github
from llama_index.llms.openai import OpenAI
from llama_index.core.tools import FunctionTool
from llama_index.core.agent import ReActAgent
from llama_index.core.agent.workflow import AgentWorkflow, FunctionAgent
from llama_index.core.workflow import Context
import asyncio
from llama_index.core.prompts import RichPromptTemplate
from llama_index.core.agent.workflow import AgentOutput, ToolCall, ToolCallResult

# Load environment variables
dotenv.load_dotenv()

# Initialize GitHub client
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
git = Github(GITHUB_TOKEN) if GITHUB_TOKEN else None

# Initialize LLM
llm = OpenAI(
    model=os.getenv("OPENAI_MODEL", "gpt-4"),
    api_key=os.getenv("OPENAI_API_KEY"),
    api_base=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
)

# Get repository information
repo_url = os.getenv("GITHUB_REPO_URL")
repo_name = repo_url.split('/')[-1].replace('.git', '')
username = repo_url.split('/')[-2]
full_repo_name = f"{username}/{repo_name}"

if git is not None:
    try:
        repo = git.get_repo(full_repo_name)
    except Exception:
        repo = None
else:
    repo = None

# State management tools
async def add_context_to_state(ctx: Context, context: str) -> Dict:
    state = await ctx.get("state")
    state["gathered_contexts"] = context
    await ctx.set("state", state)
    return {"success": True, "message": "Context added to state"}

add_context_to_state_tool = FunctionTool.from_defaults(
    fn=add_context_to_state,
    name="add_context_to_state",
    description="Adds the gathered context to the workflow state."
)

async def add_comment_to_state(ctx: Context, draft_comment: str) -> Dict:
    state = await ctx.get("state")
    state["draft_comment"] = draft_comment
    await ctx.set("state", state)
    return {"success": True, "message": "Draft comment added to state"}

add_comment_to_state_tool = FunctionTool.from_defaults(
    fn=add_comment_to_state,
    name="add_comment_to_state",
    description="Adds the draft PR comment to the workflow state."
)

# GitHub interaction tools
async def get_pr_details(pr_number: int) -> Dict:
    if not git or not repo:
        return {"error": "GitHub client not initialized"}

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
        print(f"Output from tool: {{'title': '{result['title']}', 'body': '{result['body']}'}}")
        return result
    except Exception as e:
        return {"error": str(e)}

pr_details_tool = FunctionTool.from_defaults(
    fn=get_pr_details,
    name="get_pr_details",
    description="Fetches details about a pull request."
)

async def get_file_contents(file_path: str) -> Dict:
    if not git or not repo:
        return {"error": "GitHub client not initialized"}

    try:
        file_content = repo.get_contents(file_path)
        result = {
            "content": file_content.decoded_content.decode('utf-8'),
            "path": file_content.path,
            "sha": file_content.sha,
            "size": file_content.size,
        }
        print(f"File contents retrieved: {result}")
        return result
    except Exception as e:
        return {"error": str(e)}

file_contents_tool = FunctionTool.from_defaults(
    fn=get_file_contents,
    name="get_file_contents",
    description="Fetches the contents of a file from the repository."
)

async def get_commit_details(commit_sha: str) -> Dict:
    if not git or not repo:
        return {"error": "GitHub client not initialized"}

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
            print(f"Output from tool: [{{'filename': '{f.filename}'}}]")

        result = {
            "sha": commit.sha,
            "author": commit.author.login if commit.author else None,
            "message": commit.commit.message,
            "changed_files": changed_files,
        }
        return result
    except Exception as e:
        return {"error": str(e)}

commit_details_tool = FunctionTool.from_defaults(
    fn=get_commit_details,
    name="get_commit_details",
    description="Fetches details about a commit including changed files."
)

async def add_final_review_to_state(ctx: Context, final_review: str) -> Dict:
    state = await ctx.get("state")
    state["final_review_comment"] = final_review
    await ctx.set("state", state)
    return {"success": True, "message": "Final review added to state"}

add_final_review_to_state_tool = FunctionTool.from_defaults(
    fn=add_final_review_to_state,
    name="add_final_review_to_state",
    description="Adds the final approved review comment to the workflow state."
)

async def post_review_to_github(pr_number: int, body: str) -> Dict:
    if not git or not repo:
        return {"error": "GitHub client not initialized"}

    try:
        pull_request = repo.get_pull(pr_number)
        review = pull_request.create_review(body=body)
        return {
            "success": True,
            "message": "Review posted successfully",
            "review_id": review.id,
            "html_url": review.html_url
        }
    except Exception as e:
        return {"error": str(e)}

post_review_tool = FunctionTool.from_defaults(
    fn=post_review_to_github,
    name="post_review_to_github",
    description="Posts a review comment to the specified GitHub pull request."
)

# Agents
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
    print("\nGitHub PR Review Agent Workflow")
    print("-------------------------------")
    print("Enter a PR number and request (e.g., 'Review PR #42') or 'exit' to quit:")

    ctx = Context(workflow_agent)

    while True:
        pr_number = os.environ["PR_NUMBER"]
        query = "Write a review for PR: " + pr_number
        # query = input("> ").strip()
        # if query.lower() in ('exit', 'quit'):
        #     break
        #
        # if not query:
        #     print("Please enter a valid query.")
        #     continue
        #
        # if not any(word in query.lower() for word in ["pr", "pull request", "review"]):
        #     print("Please specify that you want to review a PR (e.g., 'Review PR #42')")
        #     continue

        print("\nStarting workflow...\n")

        # Extract PR number
        import re
        pr_match = re.search(r'#?(\d+)', query)
        if pr_match:
            pr_number = int(pr_match.group(1))
            if repo:
                try:
                    pull_request = repo.get_pull(pr_number)
                    pr_body = pull_request.body if pull_request.body else ""
                    print(f"Output from tool: {{'title': '{pull_request.title}', 'body': '{pr_body}'}}")
                except Exception as e:
                    print(f"Error fetching PR directly: {e}")

        prompt = RichPromptTemplate(query)
        handler = workflow_agent.run(prompt.format(), ctx=ctx)

        current_agent = None
        async for event in handler.stream_events():
            if hasattr(event, "current_agent_name") and event.current_agent_name != current_agent:
                current_agent = event.current_agent_name
                print(f"\nCurrent agent: {current_agent}")
                # Explicit handoff logging
                print(f"Output from tool: Agent {current_agent} is now handling the request due to the following reason: {'Initial request' if current_agent == 'ReviewAndPostingAgent' else 'Requested information'}")

            elif isinstance(event, AgentOutput):
                if event.response and event.response.content:
                    print(f"\nFinal response: {event.response.content}")
                if event.tool_calls:
                    print(f"Selected tools: {[call.tool_name for call in event.tool_calls]}")
                    for call in event.tool_calls:
                        print(f"Calling selected tool: {call.tool_name}, with arguments: {call.tool_kwargs}")
                        if call.tool_name == "handoff":
                            print(f"Output from tool: Agent {call.tool_kwargs.get('to_agent')} is now handling the request due to the following reason: {call.tool_kwargs.get('reason', 'Workflow progression')}")

            elif isinstance(event, ToolCallResult):
                print(f"Output from tool: {event.tool_output}")
                if isinstance(event.tool_output, dict) and 'title' in event.tool_output:
                    print(f"Output from tool: {{'title': '{event.tool_output['title']}', 'body': '{event.tool_output.get('body', '')}'}}")

            elif isinstance(event, ToolCall):
                print(f"Calling selected tool: {event.tool_name}, with arguments: {event.tool_kwargs}")

        state = await ctx.get("state")
        print("\nCurrent workflow state:", state)
        print("\nWorkflow completed. Enter another request or 'exit' to quit.")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        print(f"Application failed: {str(e)}")
    finally:
        if git:
            git.close()