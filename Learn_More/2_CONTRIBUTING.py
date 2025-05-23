import streamlit as st

st.title("Contributing")

st.header("Origin")

st.markdown(
    """
    本專案起始於北科大城市科學實驗室之「都市數據進階分析與模型應用」課程。
    該課程目標培養學生的地理空間資訊分析、建模、視覺化及策略論述等綜合能力，
    再由實際層面的複雜都市議題切入，
    從而剖析真實環境並提出洞見。

    本專案由該課程中 Group B 組員協力製作，
    期望使用 H3 格式的地理資訊解構真實世界的資訊，
    並且結合開源資料深入淺出購房所需考量的各種面向，
    降低所有購屋族群的入門門檻。

    """
)

st.header("Artists")

st.markdown(
    """

    - 胡家禧 (台大地理系)：房屋資訊整合
    - 王思儒 (北科電機工程碩)：模型訓練
    - 賴品錕 (北科電機工程系)：「漲幅空間」指標探討
    - 鄭又榮 (中興農藝學系)：模型訓練、streamlit 網頁編寫

    """
)

st.header("Future Work")

st.markdown(
    """

    - 提升房價預測模型準確度。
    - 持續改進 UI 呈現方式以降低使用門檻。
    - 跟進時事 (小宅熱潮 - 社區規模對購屋者的影響)

    """
)