import os
import json
import pandas as pd
from datetime import date
import gradio as gr
import autogen
from autogen.cache import Cache
from finrobot.utils import get_current_date
from finrobot.data_source import FinnHubUtils, YFinanceUtils
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Utility functions
def save_output(data: pd.DataFrame, tag: str, save_path: str = None) -> None:
    if save_path:
        data.to_csv(save_path)
        print(f"{tag} saved to {save_path}")

def get_current_date():
    return date.today().strftime("%Y-%m-%d")

def register_keys():
    keys = {
        "FINNHUB_API_KEY": os.getenv("FINNHUB_API_KEY"),
        "FMP_API_KEY": os.getenv("FMP_API_KEY"),
        "SEC_API_KEY": os.getenv("SEC_API_KEY")
    }
    for key, value in keys.items():
        if value:
            os.environ[key] = value

def read_response_from_md(filename):
    with open(filename, "r") as file:
        content = file.read()
    return content

def save_to_md(content, filename):
    with open(filename, "w") as file:  # Use write mode to overwrite the file
        file.write(content + "\n")
    print(f"Content saved to {filename}")

# Initialize LLM configuration
config_list = [
    {
        "model": "gpt-4o",
        "api_key": os.getenv("OPENAI_API_KEY")
    }
]
llm_config = {"config_list": config_list, "timeout": 120, "temperature": 0}

# Register FINNHUB API keys
register_keys()

# Define agents
analyst = autogen.AssistantAgent(
    name="Market_Analyst",
    system_message="As a Market Analyst, one must possess strong analytical and problem-solving abilities, collect necessary financial information and aggregate them based on client's requirement. For coding tasks, only use the functions you have been provided with. Reply TERMINATE when the task is done.",
    llm_config=llm_config,
)

user_proxy = autogen.UserProxyAgent(
    name="User_Proxy",
    is_termination_msg=lambda x: x.get("content", "") and x.get("content", "").strip().endswith("TERMINATE"),
    human_input_mode="NEVER",
    max_consecutive_auto_reply=10,
    code_execution_config={
        "work_dir": "coding",
        "use_docker": False,
    },
)

# Register tools
from finrobot.toolkits import register_toolkits

tools = [
    {
        "function": FinnHubUtils.get_company_profile,
        "name": "get_company_profile",
        "description": "get a company's profile information"
    },
    {
        "function": FinnHubUtils.get_company_news,
        "name": "get_company_news",
        "description": "retrieve market news related to designated company"
    },
    {
        "function": FinnHubUtils.get_basic_financials,
        "name": "get_financial_basics",
        "description": "get latest financial basics for a designated company"
    },
    {
        "function": YFinanceUtils.get_stock_data,
        "name": "get_stock_data",
        "description": "retrieve stock price data for designated ticker symbol"
    }
]
register_toolkits(tools, analyst, user_proxy)

def save_response_to_json(response, filename):
    response_dict = {
        "chat_id": response.chat_id,
        "chat_history": response.chat_history,
        "summary": response.summary,
        "cost": response.cost,
        "human_input": response.human_input
    }
    with open(filename, "w") as file:
        file.write(json.dumps(response_dict, indent=4))
    print(f"Response saved to {filename}")

# Function to initiate chat and save response
def initiate_chat_and_save_response(analyst, user_proxy, company):
    today_date = get_current_date()
    json_filename = f"result_{company}_{today_date}.json"
    md_filename = f"result_{company}_{today_date}.md"
    
    # Check if MD file already exists
    if os.path.exists(md_filename):
        return read_response_from_md(md_filename)
    
    with Cache.disk() as cache:
        response = user_proxy.initiate_chat(
            analyst,
            message=f"Report in Chinese. Use all the tools provided to retrieve information available for {company} upon {get_current_date()}. Analyze the positive developments and potential concerns of {company} with 2-4 most important factors respectively and keep them concise. Most factors should be inferred from company related news. Then make a rough prediction (e.g. up/down by %) of the {company} stock price movement for next week. Provide a summary analysis to support your prediction.",
            cache=cache,
        )
        
        save_response_to_json(response, json_filename)
        return json.dumps(response.chat_history, indent=4)

def filter_user_content(chat_history):
    # 解析 chat_history 为 JSON 对象
    chat_history_dict = json.loads(chat_history)
    # 查找用户需要的内容
    for entry in chat_history_dict:
        if entry['role'] == 'user' and "###" in entry['content']:
            return entry['content']
    return "No relevant content found."

# 使用更新的函数在 analyze_company 中
def analyze_company(company):
    if company:
        company = company.upper()
        today_date = get_current_date()
        md_filename = f"result_{company}_{today_date}.md"
        
        # Check if MD file already exists
        if os.path.exists(md_filename):
            return read_response_from_md(md_filename)
        
        content = initiate_chat_and_save_response(analyst, user_proxy, company)
        # 筛选有效的用户内容
        filtered_content = filter_user_content(content)
        save_to_md(filtered_content, md_filename)  # 只保存筛选后的内容
        return filtered_content

# 自定义CSS样式
custom_css = """
h1, h2, h3, h4, h5, h6 {
    font-family: 'Arial', sans-serif;
    font-weight: bold;
}
body {
    font-family: 'Arial', sans-serif;
}
.gradio-container {
    max-width: 800px;
    margin: auto;
    padding: 20px;
    border: 1px solid #ccc;
    border-radius: 10px;
    box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
}
textarea, input, .btn-primary {
    font-size: 16px !important;
    padding: 10px !important;
    border-radius: 5px !important;
}
#component-0 > .wrap > .block.markdown-block > .markdown {
    font-size: 24px !important;
    line-height: 1.8 !important;
}
"""

iface = gr.Interface(
    fn=analyze_company,
    inputs=gr.Textbox(lines=1, placeholder="输入英文公司名称或股票代码"),
    outputs=gr.Markdown(label="Trade-Helper 股市基本面分析与预测"),
    title="Trade-Helper",
    description="请输入公司名称或股票代码，以获取AI的分析和预测报告。",
    css=custom_css,
    allow_flagging='never'
)

if __name__ == "__main__":
    iface.launch(share=True)
