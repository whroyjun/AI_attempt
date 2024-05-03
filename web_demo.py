#!/usr/bin/env python
# coding=utf-8
import gradio as gr
import nltk
from chinaai_utils import get_completion
from chinaai_utils import get_embedding
from emmbedding_query_utils import get_embedding_query
from prompt_utils import build_prompt
from vectordb_utils import InMemoryVecDB
from pdf_utils import extract_text_from_pdf
from text_utils import split_text

#already download into local
#nltk.download('punkt')
vec_db = InMemoryVecDB("demo_ernie",embedding_fn=get_embedding,embedding_fn_query=get_embedding_query)


def init_db(file):
    #paragraphs = extract_text_from_pdf(file.name)
    paragraphs = extract_text_from_pdf(file.name,page_numbers=[2,3],min_line_length=10)
    #documents_pdf = split_text(paragraphs, 500, 100)
    documents =paragraphs
    print("###document upload")
    """
    documents = [
    "联合国就苏丹达尔富尔地区大规模暴力事件发出警告",
    "土耳其、芬兰、瑞典与北约代表将继续就瑞典“入约”问题进行谈判",
    ]
    """
    vec_db.add_documents(documents)


def chat(user_input, chatbot, context, search_field):
    search_results = vec_db.search(user_input, 2)
    print("search in local DB completed with results",search_results)
    search_field = "\n\n".join(search_results)
    prompt = build_prompt(info=search_results, query=user_input)
    #response = get_completion(prompt, context)
    response = get_completion(prompt)
    chatbot.append((user_input, response))
    context.append({'role': 'user', 'content': user_input})
    context.append({'role': 'assistant', 'content': response})
    return "", chatbot, context,search_field


def reset_state():
    return [], [], "", ""


def main():
    with gr.Blocks() as demo:
        gr.HTML("""<h1 align="center">PnG_AI_Access_Management</h1>""")

        with gr.Row():
            with gr.Column():
                fileCtrl = gr.File(label="上传文件", file_types=[',pdf'])

        with gr.Row():
            with gr.Column(scale=2):
                chatbot = gr.Chatbot()
            with gr.Column(scale=2):
                # gr.HTML("""<h4>检索结果</h4>""")
                search_field = gr.Textbox(show_label=False, placeholder="检索结果...", lines=10)
                user_input = gr.Textbox(show_label=False, placeholder="输入框...", lines=2)
                #user_input='How many parameters does llama2 have?'
                with gr.Row():
                    submitBtn = gr.Button("提交", variant="primary")
                    emptyBtn = gr.Button("清空")

        context = gr.State([])

        submitBtn.click(chat, [user_input, chatbot, context, search_field],
                        [user_input, chatbot, context, search_field])
        emptyBtn.click(reset_state, outputs=[chatbot, context, user_input, search_field])

        fileCtrl.upload(init_db, inputs=[fileCtrl])

    demo.queue().launch(share=False, server_name='0.0.0.0', server_port=8888, inbrowser=True)


if __name__ == "__main__":
    main()
