import streamlit as st

elif st.session_state["mode"] == "chat":
    st.subheader("💬 Chat History")
    if st.button("⬅️ Back to Home"):
        st.session_state["mode"] = "home"
        st.rerun()
    st.markdown("---")
    for role, message in st.session_state["chat"]:
        if role == "user":
            st.markdown(f"<div class='user-bubble'><b>🧑 You:</b><br>{message}</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='bot-bubble'><b>🤖 AI:</b><br>{message}</div>", unsafe_allow_html=True)
    st.markdown("<div id='chat-end'></div>", unsafe_allow_html=True)
