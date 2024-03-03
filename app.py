import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

from llama_index import VectorStoreIndex, SimpleDirectoryReader
from llama_index import ServiceContext
from llama_index import LLMPredictor
from typing import Optional, List, Mapping, Any
from transformers import pipeline
from llama_index import ServiceContext, SimpleDirectoryReader, SummaryIndex
from llama_index.callbacks import CallbackManager
from llama_index.llms import (
    CustomLLM,
    CompletionResponse,
    CompletionResponseGen,
    LLMMetadata,
)
from llama_index.llms.base import llm_completion_callback
from transformers import AutoModelForCausalLM, AutoTokenizer
from llama_index.embeddings import resolve_embed_model
import streamlit as st


@st.cache_resource
def get_model():
    print("Loading qwen model...")
    # 使用huggingface库加载Qwen-7B-Chat模型 https://huggingface.co/Qwen/Qwen-7B-Chat
    device = "cuda" # the device to load the model onto
    # Model names: "Qwen/Qwen-7B-Chat", "Qwen/Qwen-14B-Chat"
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen-7B-Chat", trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen-7B-Chat",
        device_map="auto",
        trust_remote_code=True
    ).eval()
    print("Model loading completed!...")
    # 指定本地嵌入模型路径
    print("Loading embedding model...")
    local_embed_model_path = "/home/hujili/code/model/m3e-base"
    # 解析本地嵌入模型
    embed_model = resolve_embed_model(f"local:{local_embed_model_path}")
    print("Embedding model loading completed!...")
    return tokenizer, model, embed_model

tokenizer, model, embed_model = get_model()

class OurLLM(CustomLLM):
    context_window: int = 2048
    num_output: int = 256
    model_name: str = "Qwen-7B-Chat"
    @property
    def metadata(self) -> LLMMetadata:
        """Get LLM metadata."""
        return LLMMetadata(
            context_window=self.context_window,
            num_output=self.num_output,
            model_name=self.model_name,
        )

    @llm_completion_callback()
    def complete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        prompt_length = len(prompt)

        # only return newly generated tokens
        text,_ = model.chat(tokenizer, prompt, history=[])
        return CompletionResponse(text=text)

    @llm_completion_callback()
    def stream_complete(
        self, prompt: str, **kwargs: Any
    ) -> CompletionResponseGen:
        raise NotImplementedError()
    
@st.cache_resource(show_spinner=False)  # type: ignore[misc]
def load_index() -> Any:
    print("Loading index...")
    llm = OurLLM()
    service_context = ServiceContext.from_defaults(llm=llm,embed_model=embed_model)
    documents = SimpleDirectoryReader("./data").load_data()
    index = VectorStoreIndex.from_documents(documents,service_context=service_context)
    print("Index loading completed!")
    #构建查询引擎
    query_engine = index.as_query_engine()
    return query_engine

def main() -> None:
    """运行 JavaCopilot"""
    
    if "query_engine" not in st.session_state:
        st.session_state.query_engine = load_index()

    st.title("JavaCopilot")
    if "messages" not in st.session_state:
        system_prompt = (
            "作为Java编程专家，您的任务是根据所提供的上下文文档为用户问题提供答案。 如果问题超出了文档的范围，请回复“未找到相关信息"
            "您的回答应该准确、相关并与上下文文档的内容保持一致。 能够处理与 Java 编程相关的各种问题，确保提供的信息准确且对用户有帮助 "
            "请使用中文回答问题"
        )
        st.session_state.messages = [{"role": "system", "content": system_prompt}]

    for message in st.session_state.messages:
        if message["role"] not in ["user", "assistant"]:
            continue
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input():
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            print("正在调用查询引擎API...")
            response = st.session_state.query_engine.query(prompt)
            full_response = f"{response}"
            print(full_response)
            message_placeholder.markdown(full_response)
            st.session_state.messages.append({"role": "assistant", "content": full_response})


if __name__ == "__main__":
    main()
