from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
import streamlit as st
import json

INDEX_NAME='trade-finance-dense-db'
# NAMESPACE='default'
NAMESPACES= ['default', 'LND_HISTORY']
TOP_K_VAL=3

META_KEYS=["FACILITY_CODE", "FACILITY_ORDER", "PROCESS_TYPE", "HISTORY_SEQ_NO", "JOURNAL_NO", "LOAN_KEY", "SOURCE_TYPE"]

def get_embeddings(text):
    pc = Pinecone(st.secrets["api_key"])
    res = pc.inference.embed(
        model="multilingual-e5-large",
        inputs=text,
        parameters={
            "input_type": "query",
            "truncate": "END"
        }
    )

    embedding = res.data[0]['values']
    return embedding

def query(question):
    pc = Pinecone(st.secrets["api_key"])
    index = pc.Index(INDEX_NAME)
    model=SentenceTransformer('intfloat/multilingual-e5-large')
    query_vector = model.encode(question)
    results = index.query_namespaces(
        namespaces=NAMESPACES,
        vector=query_vector.tolist(),
        top_k=6,
        metric="cosine",
        include_metadata=True,
        filter={
            "FACILITY_CODE": {
                "$eq": "LTC-FCY"
            }
        }
    )

    jsonArr = []
    for sv in results['matches']:
        jsObj = json.dumps(sv.to_dict(), indent=2)
        print(f"Type of jsObj: {type(jsObj)}")
        print(jsObj)
        jsonStr = json.dumps(jsObj, indent=4, ensure_ascii=False)
        jsonArr.append(jsonStr)

    return jsonArr

def show_ui():
    st.title("Search Data of Limit Facility")
    st.header("Ask questions about the old limit facility data")

    st.subheader("Add Search Filters")
    if "filters" not in st.session_state:
        st.session_state.filters = []

    def add_filter():
        st.session_state.filters.append({"search_by": "FACILITY_CODE", "value": ""})

    if st.button("Add Filter"):
        add_filter()

    # Render dynamic filters
    for idx, filter_item in enumerate(st.session_state.filters):
        cols = st.columns([2, 3, 1])
        with cols[0]:
            search_by = st.selectbox(
                "Search by",
                options=META_KEYS,
                index=META_KEYS.index(filter_item["search_by"]),
                key=f"search_by_{idx}"
            )
        with cols[1]:
            value = st.text_input(
                f"Enter value for {search_by}",
                value=filter_item["value"],
                key=f"value_{idx}"
            )
        with cols[2]:
            if st.button("Remove", key=f"remove_{idx}"):
                st.session_state.filters.pop(idx)
                st.experimental_rerun()
        # Update filter values in session state
        st.session_state.filters[idx]["search_by"] = search_by
        st.session_state.filters[idx]["value"] = value

    # Build filter_query from all filters
    filter_query = {}
    for f in st.session_state.filters:
        if f["value"]:
            filter_query[f["search_by"]] = {"$eq": f["value"]}

    # For backward compatibility, if no filters, show default single filter
    if not st.session_state.filters:
        search_by = st.selectbox(
            "Search by",
            options=META_KEYS,
            index=0,
            key="default_search_by"
        )
        value = st.text_input(f"Enter value for {search_by}", key="default_value")
        if value:
            filter_query = {search_by: {"$eq": value}}

    question = st.text_area("Or enter your query (multi-line)", height=100)

    top_k = st.number_input(
        "Top K Results",
        min_value=1,
        max_value=20,
        value=TOP_K_VAL,
        step=1
    )

    if st.button("Search"):
        with st.spinner("Searching..."):
            filter_query = {search_by: {"$eq": value}} if value else {}
            pc = Pinecone(st.secrets["api_key"])
            index = pc.Index(INDEX_NAME)
            model = SentenceTransformer('intfloat/multilingual-e5-large')
            query_vector = model.encode(question if question else "search")
            results = index.query_namespaces(
                namespaces=NAMESPACES,
                vector=query_vector.tolist(),
                top_k=top_k,
                metric="cosine",
                include_metadata=True,
                filter=filter_query
            )
            if results['matches']:
                for idx, jsObj in enumerate(results.matches):
                    # jsObj = json.dumps(sv.to_dict(), indent=2)
                    st.markdown(f"**Result {idx+1}:**")
                    st.markdown(f"```json\n{jsObj}\n```")
            else:
                st.info("No results found.")

def show_chat_ui():
    st.title("Search Data of Limit Facility")
    st.header("Ask questions about the old limit facility data")
    
    if "messages" not in st.session_state:
        st.session_state.messages = []
        st.session_state.chat_history = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    if prompt := st.chat_input("Ask a question about the limit facility..."):
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.spinner("Working on your query..."):
            response = query(question=prompt)
            with st.chat_message("assistant"):
                for idx, item in enumerate(response, 1):
                    st.markdown(f"**Result {idx}:**")
                    st.markdown(f"```json\n{item}\n```")
                # st.markdown(response)

            st.session_state.messages.append({"role": "user", "content": prompt})
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.session_state.chat_history.extend([(prompt, response)])

if __name__ == '__main__':
    show_ui()
    # query("Facility has not been modified since creation and its facility code is 'LTC-FCY'")