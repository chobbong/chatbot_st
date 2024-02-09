import os
import json
from tqdm import tqdm

from pinecone import Pinecone
from langchain.vectorstores import pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain

import streamlit as st
import streamlit_chat


PINECONE_INDEX_NAME = 'llm-rag-openai' # 파인콘 인덱스 이름
EMBEDDER_NAME = 'text-embedding-ada-002'
LLM_NAME = 'gpt-3.5-turbo-0125'


if __name__ == '__main__':
    # 시작 준비
    with tqdm(total=100, ncols=100, leave=True) as pbar:
        # 임베더 (openai ada 002)
        pbar.set_description("임베더 불러오는중..")
        embedder = OpenAIEmbeddings(
            model=EMBEDDER_NAME,
            openai_api_key=os.getenv('OPENAI_KEY')
        )
        pbar.update(25)

        # 파인콘 연결
        pbar.set_description("파인콘 연결중..")
        pc = Pinecone(
            api_key=os.getenv('PINECONE_API_KEY'),
            environment=os.getenv('PINECONE_API_ENV')
        )
        pc_index = pc.Index(PINECONE_INDEX_NAME)
        pbar.update(25)

        # 랭체인과 파인콘 연동
        pbar.set_description("파인콘&랭체인 연동중..")
        vectorstore = pinecone.Pinecone(
            index=pc_index,
            embedding=embedder,
            text_key='text'
        )
        pbar.update(25)

        # 언어모델 (gpt 3.5)
        pbar.set_description("LLM 로딩중..")
        llm = ChatOpenAI(
            api_key=os.getenv("OPENAI_KEY"),
            temperature=0.1,
            max_tokens=512,
            model_name=LLM_NAME
        )
        qa_chain = load_qa_chain(llm, chain_type='stuff')
        pbar.update(25)

        print("#######준비완료#######")
        print(f"임베딩모델 : {EMBEDDER_NAME}\nLLM : {LLM_NAME}")

    # llm 텍스트생성
    def llm_gen(input_text:str):
        similar_chunks = vectorstore.similarity_search(input_text, k=10)
        case_nums = {i.metadata['case_num'] for i in similar_chunks}
        
        output_text = qa_chain.run(input_documents=similar_chunks, question=input_text)
        return output_text, case_nums
    
    # bot 응답생성
    def bot_respond(input_text):
        output_text, case_nums = llm_gen(input_text)
        output_text += '\n\n참고판례 : ' + str(", ".join([i for i in case_nums]))
        return output_text
    
    # Streamlit UI 설정
    if 'generated' not in st.session_state:
        st.session_state['generated'] = []
    if 'past' not in st.session_state:
        st.session_state['past'] = []

    with st.form('form', clear_on_submit=True):
        user_input = st.text_input('판례에 관해 물어보세요.', '', key='input')
        submitted = st.form_submit_button('Send')

    if submitted and user_input:
        st.session_state['past'].append(user_input)
        # 입력
        chatbot_response = bot_respond(user_input)
        st.session_state['generated'].append(chatbot_response)

    # 사용자의 질문과 첫봇의 답변을 순차적으로 화면에 출력
    if st.session_state['generated']:
        for i in reversed(range(len(st.session_state['generated']))):
            streamlit_chat.message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')
            streamlit_chat.message(st.session_state['generated'][i], key=str(i))
