import re
import os
import json
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from openai import OpenAI
from utils.data_tools import load_data

st.title("📞 CareLine")

with st.sidebar:
    st.title("📞 Hotline Operator Assistant")
    st.markdown("## 📄 Documentation")
    st.markdown("""**Hotline Operator Assistant** is an AI agent who assist operator during the time of whistleblower's cases reporting.
    The Operator can ask the Agent for information and statistic of the cases as follows:"""
    )
    st.markdown("""**1. ℹ️ Information of the cases**: the Operator can ask the information of the case by providing case number to the Agent.
    The Agent will retrieve the information of the case and list the detail of the case for the Operator.
    """)
    st.markdown("""**2. 📊 Statistic of the cases**: the Operator and stakeholder can ask the Agent to provide statistic and insight of reporting via hotline.
    The Agent will provide you a visualization of the cases reported.
    """)
    st.markdown("## ⚠️ Remark")
    st.markdown("The Agent solely relies on the **hotline log**. You may get a made up information if you ask anything that is not in database or hotline log.")

system_prompt = f"""You are 📞CareLine, a Hotline Operator Asisstant. Your task is to provide assistance to the hotline operator. \
    You are given tools as below to retrieve information from database: 
        - get_info: use this tool to retrieve information of the case from database.
        - generate_chart: use this tool to visualize and create chart based on user instruction

    The rule are as follows:
        - If the operators ask for information of the case, you should provide feedback of the case to operator without modifying the content. \
        - If the tool is not necessary, answer the user question as usual.
        - If operator ask for Metro customer service contact information provide 02-617-6000 to operator
"""

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "system", "content": system_prompt},
        {"role": "assistant", "content": "Hello! How can I help you?"}
    ]

client = OpenAI(
    api_key = st.secrets["TYPHOON_API_KEY"],
    base_url = st.secrets["TYPHOON_BASE_URL"]
)

df = load_data("../data/hotline_logv3.xlsx")

schema = f"""
CallId: Unique name of report
Company: Company which whistleblower is working
CaseNumber: Unique number of case reported
Coverage: Column indicating if the case is out of scope or not
ReportDate: Date of case reported
ReportTime: Time of case reported
Channel: Channel of case reported (e.g. {", ".join([channel for channel in df["Channel"].unique().tolist() if channel is not np.nan])})
Operator: Operator receiving a case reported
WhistleblowerType: Column specifying if the Whistleblower is employee of the company or third party
Identity: Column indicating if the Whistleblower disclose their identity or not
IncidentType: Column indicating type of incident such as 'Inappropriate behavior'
AllegedPerson: AllegedPerson indicating identity of aleged person such as name
SubmissionDate: Date of report submitted to the Company's management
"""

function_definition = [
    {
        "type": "function",
        "function": {
            "name": "get_info",
            "description": "Get information of the case based on given case number",
            "parameters": {
                "type": "object",
                "properties": {
                    "case_number": {
                        "type": "string",
                        "description": "case number of the report"
                    }
                }
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "chart_generator",
            "description": "Generate chart and return python code based on user instruction or question",
            "parameters": {
                "type": "object",
                "properties": {
                    "user_instruction": {
                        "type": "string",
                        "description": "user instruction on creating chart and visualization"
                    }
                }
            }
        }
    }
]

def get_info(case_number: str) -> str:
    case_info = df[df["CaseNumber"] == case_number]
    case_info_string = []

    for col in case_info.columns:
        info_string = f"{col}: {case_info[col].values}"
        case_info_string.append(info_string)

    if len(case_info_string) > 0:
        return "\n".join(case_info_string)
    else:
        return "No case reported"

def chart_generator(user_instruction):
    instruction = f"""You are visualization expert.
    Return the answer in this format.

    <execute_python>
    # valid python code here
    </execute_python>

    Do not add explanations, only the tags and code

    The code should create a visualization from a DataFrame 'df' with these columns:
        {schema}

        Requirements for the code:
        1. Assume the DataFrame is already loaded as 'df'.
        2. Use matplotlib for plotting.
        3. Use subplots to create chart
        3. Add clear title, axis labels, and legend if needed.
        4. Add all necessary import python statements
        5. Do not call plt.show()
        6. Close all plots with plt.close()
        7. Assume the DataFrame is already in prepared and cleaned
        8. Do not add grid in chart
        9. Always place legend at top right corner
        10. Always use subplots with vairable fig and ax
    """

    response = client.chat.completions.create(
        model = st.secrets["TYPHOON_MODEL"],
        messages = [
            {"role": "system", "content": instruction},
            {"role": "user", "content": user_instruction}
        ]
    )

    python_code = response.choices[0].message.content
    python_code = re.search(r"<execute_python>([\s\S]*?)</execute_python>", python_code).group(1).strip()

    local_scope = {}
    exec(python_code, {"df": df}, local_scope)

    return local_scope["fig"]

def get_response():

    messages = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages if m["role"] != "chart"]

    response = client.chat.completions.create(
        model = st.secrets["TYPHOON_MODEL"],
        messages = messages,
        tools = function_definition
    )

    tool_calls = response.choices[0].message.tool_calls

    if tool_calls:
        for tool_call in tool_calls:
            function_name = tool_call.function.name
            arguments = tool_call.function.arguments

            if function_name == "chart_generator":
                kwargs = json.loads(arguments)
                result = chart_generator(**kwargs)

            return result

    return response.choices[0].message.content

for m in st.session_state.messages:
    if m["role"] not in ["system", "tool"]:
        if m["role"] != "chart":
            with st.chat_message(m["role"]):
                st.markdown(m["content"])
        else:
            with st.chat_message("assistant"):
                st.pyplot(m["content"])

if prompt := st.chat_input("Say Something"):
    with st.chat_message("user"):
        st.markdown(prompt)

    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        with st.spinner():
            response = get_response()

            if isinstance(response, Figure):
                st.markdown("Please find chart as requested:")
                st.pyplot(response)
                st.session_state.messages.append({"role": "chart", "content": response})
            else:
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})