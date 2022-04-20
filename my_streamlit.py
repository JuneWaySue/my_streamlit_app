import json
import time
import random
import datetime

import requests
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from streamlit_echarts import st_echarts

from streamlit.server.server import Server
# from streamlit.script_run_context import get_script_run_ctx as get_report_ctx
from streamlit.scriptrunner import get_script_run_ctx as get_report_ctx

import graphviz
import pydeck as pdk
import altair as alt
import plotly.figure_factory as ff
import matplotlib.pyplot as plt
from pyecharts.charts import *
from pyecharts.globals import ThemeType
from pyecharts import options as opts
from pyecharts.commons.utils import JsCode

from PIL import Image
from io import BytesIO


def main():
    st.set_page_config(page_title="七里香还是稻香",page_icon=":rainbow:",layout="wide",initial_sidebar_state="auto")
    st.title('七里香还是稻香:heart:')
    st.markdown('<br>',unsafe_allow_html=True)
    st.markdown('<br>',unsafe_allow_html=True)
    charts_mapping={
        'Line':'line_chart','Bar':'bar_chart','Area':'area_chart','Hist':'pyplot','Altair':'altair_chart',
        'Map':'map','Distplot':'plotly_chart','Pdk':'pydeck_chart','Graphviz':'graphviz_chart','PyEchart':''
    }
    if 'first_visit' not in st.session_state:
        st.session_state.first_visit=True
    else:
        st.session_state.first_visit=False
    # 初始化全局配置
    if st.session_state.first_visit:
        # 在这里可以定义任意多个全局变量，方便程序进行调用
        st.session_state.date_time=datetime.datetime.now() + datetime.timedelta(hours=8) # Streamlit Cloud的时区是UTC，加8小时即北京时间
        st.session_state.random_chart_index=random.choice(range(len(charts_mapping)))
        st.session_state.my_random=MyRandom(random.randint(1,1000000))
        st.session_state.city_mapping,st.session_state.random_city_index=get_city_mapping()
        # st.session_state.random_city_index=random.choice(range(len(st.session_state.city_mapping)))
        st.balloons()
        st.snow()

    music=st.sidebar.radio('Select Music You Like',['七里香','稻香'],index=random.choice(range(2)))
    st.sidebar.write(f'正在播放 {music}-周杰伦 :musical_note:')
    audio_bytes=get_audio_bytes(music)
    st.sidebar.audio(audio_bytes, format='audio/mp3')

    d=st.sidebar.date_input('Date',st.session_state.date_time.date())
    t=st.sidebar.time_input('Time',st.session_state.date_time.time())
    t=f'{t}'.split('.')[0]
    st.sidebar.write(f'The current date time is {d} {t}')
    chart=st.sidebar.selectbox('Select Chart You Like',charts_mapping.keys(),index=st.session_state.random_chart_index)
    city=st.sidebar.selectbox('Select City You Like',st.session_state.city_mapping.keys(),index=st.session_state.random_city_index)
    color = st.sidebar.color_picker('Pick A Color You Like', '#520520')
    st.sidebar.write('The current color is', color)

    with st.container():
        st.markdown(f'### {city} Weather Forecast')
        forecastToday,df_forecastHours,df_forecastDays=get_city_weather(st.session_state.city_mapping[city])
        col1,col2,col3,col4,col5,col6=st.columns(6)
        col1.metric('Weather',forecastToday['weather'])
        col2.metric('Temperature',forecastToday['temp'])
        col3.metric('Body Temperature',forecastToday['realFeel'])
        col4.metric('Humidity',forecastToday['humidity'])
        col5.metric('Wind',forecastToday['wind'])
        col6.metric('UpdateTime',forecastToday['updateTime'])
        c1 = (
            Line()
            .add_xaxis(df_forecastHours.index.to_list())
            .add_yaxis('Temperature', df_forecastHours.Temperature.values.tolist())
            .add_yaxis('Body Temperature', df_forecastHours['Body Temperature'].values.tolist())
            .set_global_opts(
                title_opts=opts.TitleOpts(title="24 Hours Forecast"),
                xaxis_opts=opts.AxisOpts(type_="category"),
                yaxis_opts=opts.AxisOpts(type_="value",axislabel_opts=opts.LabelOpts(formatter="{value} °C")),
                tooltip_opts=opts.TooltipOpts(trigger="axis", axis_pointer_type="cross")
                )
            .set_series_opts(label_opts=opts.LabelOpts(formatter=JsCode("function(x){return x.data[1] + '°C';}")))
        )

        c2 = (
            Line()
            .add_xaxis(xaxis_data=df_forecastDays.index.to_list())
            .add_yaxis(series_name="High Temperature",y_axis=df_forecastDays.Temperature.apply(lambda x:int(x.replace('°C','').split('~')[1])))
            .add_yaxis(series_name="Low Temperature",y_axis=df_forecastDays.Temperature.apply(lambda x:int(x.replace('°C','').split('~')[0])))
            .set_global_opts(
                title_opts=opts.TitleOpts(title="7 Days Forecast"),
                xaxis_opts=opts.AxisOpts(type_="category"),
                yaxis_opts=opts.AxisOpts(type_="value",axislabel_opts=opts.LabelOpts(formatter="{value} °C")),
                tooltip_opts=opts.TooltipOpts(trigger="axis", axis_pointer_type="cross")
                )
            .set_series_opts(label_opts=opts.LabelOpts(formatter=JsCode("function(x){return x.data[1] + '°C';}")))
        )

        t = Timeline(init_opts=opts.InitOpts(theme=ThemeType.LIGHT,width='1200px'))
        t.add_schema(play_interval=10000,is_auto_play=True)
        t.add(c1, "24 Hours Forecast")
        t.add(c2, "7 Days Forecast")
        components.html(t.render_embed(), width=1200, height=520)
        with st.expander("24 Hours Forecast Data"):
            st.table(df_forecastHours.style.format({'Temperature':'{}°C','Body Temperature':'{}°C','Humidity':'{}%'}))
        with st.expander("7 Days Forecast Data",expanded=True):
            st.table(df_forecastDays)

    st.markdown(f'### {chart} Chart')
    df=get_chart_data(chart,st.session_state.my_random)
    eval(f'st.{charts_mapping[chart]}(df{",use_container_width=True" if chart in ["Distplot","Altair"] else ""})' if chart != 'PyEchart' else f'st_echarts(options=df)')

    st.markdown('### Animal Pictures')
    pictures=get_pictures(st.session_state.my_random)
    if pictures:
        col=st.columns(len(pictures))
        for idx,(name,img) in enumerate(pictures.items()):
            eval(f"col[{idx}].image(img, caption=name,use_column_width=True)")
    else:
        st.warning('Get pictures fail.')

    st.markdown('### Some Ads Videos')
    session_id = get_report_ctx().session_id
    sessions = Server.get_current()._session_info_by_id
    session_ws = sessions[session_id].ws
    st.sidebar.info(f'当前在线人数：{len(sessions)}')
    col1,col2=st.columns(2)
    video1,video2=get_video_bytes()
    col1.video(video1, format='video/mp4', start_time=2)
    col2.video(video2, format='video/mp4')
    # if session_ws is not None:
    #     session_headers = session_ws.request.headers
    #     if 'Windows' not in session_headers['User-Agent']:
    #         st.info(f"Sorry!!!  \nOnly show videos on PC")
    #     else:
    #         col1,col2=st.columns(2)
    #         video1=get_video_bytes('开不了口')
    #         col1.video(video1)
    #         video2=get_video_bytes('最长的电影')
    #         col2.video(video2, format='video/mp4')
    # else:
    #     st.info('Please refresh the page.')

    st.markdown('<br>',unsafe_allow_html=True)
    st.markdown('<br>',unsafe_allow_html=True)
    st.markdown('### About Me')
    with open('README.md','r') as f:
        readme=f.read()
    st.markdown(readme)

    st.markdown('<br>',unsafe_allow_html=True)
    st.markdown('<br>',unsafe_allow_html=True)
    with st.expander("View Code"):
        with open('my_streamlit.py','r') as f:
            code=f.read()
        st.code(code,language="python")

class MyRandom:
    def __init__(self,num):
        self.random_num=num

def my_hash_func(my_random):
    num = my_random.random_num
    return num

@st.cache(hash_funcs={MyRandom: my_hash_func},allow_output_mutation=True,ttl=3600)
def get_chart_data(chart,my_random):
    data=np.random.randn(20,3)
    df=pd.DataFrame(data,columns=['a', 'b', 'c'])
    if chart in ['Line','Bar','Area']:
        return df

    elif chart == 'Hist':
        arr = np.random.normal(1, 1, size=100)
        fig, ax = plt.subplots()
        ax.hist(arr, bins=20)
        return fig

    elif chart == 'Altair':
        df = pd.DataFrame(np.random.randn(200, 3),columns=['a', 'b', 'c'])
        c = alt.Chart(df).mark_circle().encode(x='a', y='b', size='c', color='c', tooltip=['a', 'b', 'c'])
        return c

    elif chart == 'Map':
        df = pd.DataFrame(np.random.randn(1000, 2) / [50, 50] + [37.76, -122.4],columns=['lat', 'lon'])
        return df

    elif chart == 'Distplot':
        x1 = np.random.randn(200) - 2
        x2 = np.random.randn(200)
        x3 = np.random.randn(200) + 2
        # Group data together
        hist_data = [x1, x2, x3]
        group_labels = ['Group 1', 'Group 2', 'Group 3']
        # Create distplot with custom bin_size
        fig = ff.create_distplot(hist_data, group_labels, bin_size=[.1, .25, .5])
        # Plot!
        return fig

    elif chart == 'Pdk':
        df = pd.DataFrame(np.random.randn(1000, 2) / [50, 50] + [37.76, -122.4],columns=['lat', 'lon'])
        args=pdk.Deck(map_style='mapbox://styles/mapbox/light-v9',
            initial_view_state=pdk.ViewState(latitude=37.76,longitude=-122.4,zoom=11,pitch=50,),
            layers=[pdk.Layer('HexagonLayer',data=df,get_position='[lon, lat]',radius=200,elevation_scale=4,elevation_range=[0, 1000],pickable=True,extruded=True),
            pdk.Layer('ScatterplotLayer',data=df,get_position='[lon, lat]',get_color='[200, 30, 0, 160]',get_radius=200)])
        return args

    elif chart == 'Graphviz':
        graph = graphviz.Digraph()
        graph.edge('grandfather', 'father')
        graph.edge('grandmother', 'father')
        graph.edge('maternal grandfather', 'mother')
        graph.edge('maternal grandmother', 'mother')
        graph.edge('father', 'brother')
        graph.edge('mother', 'brother')
        graph.edge('father', 'me')
        graph.edge('mother', 'me')
        graph.edge('brother', 'nephew')
        graph.edge('Sister-in-law', 'nephew')
        graph.edge('brother', 'niece')
        graph.edge('Sister-in-law', 'niece')
        graph.edge('me', 'son')
        graph.edge('me', 'daughter')
        graph.edge('where my wife?', 'son')
        graph.edge('where my wife?', 'daughter')
        return graph

    elif chart == 'PyEchart':
        options = {
            "xAxis": {
                "type": "category",
                "data": ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"],
            },
            "yAxis": {"type": "value"},
            "series": [
                {"data": [820, 932, 901, 934, 1290, 1330, 1320], "type": "line"}
            ],
        }
        return options

@st.cache(hash_funcs={MyRandom: my_hash_func},suppress_st_warning=True,ttl=3600)
def get_pictures(my_random):
    def _get_one(url,what):
        try:
            img=Image.open(BytesIO(requests.get(requests.get(url).json()[what]).content))
            return img
        except Exception as e:
            if 'cannot identify image file' in str(e):
                return _get_one(url,what)
        return False
    imgs={}
    mapping={
        'https://aws.random.cat/meow':{'name':'A Cat Picture','what':'file'},
        'https://random.dog/woof.json':{'name':'A Dog Picture','what':'url'},
        'https://randomfox.ca/floof/':{'name':'A Fox Picture','what':'image'},
    }
    for url,url_map in mapping.items():
        img=_get_one(url,url_map['what'])
        if img:
            imgs[url_map['name']]=img

    return imgs

@st.cache(ttl=3600)
def get_city_mapping():
    url='https://h5ctywhr.api.moji.com/weatherthird/cityList'
    r=requests.get(url)
    data=r.json()
    city_mapping=dict()
    guangzhou=0
    flag=True
    for i in data.values():
        for each in i:
            city_mapping[each['name']]=each['cityId']
            if each['name'] != '广州市' and flag:
                guangzhou+=1
            else:
                flag=False

    return city_mapping,guangzhou

@st.cache(ttl=3600)
def get_city_weather(cityId):
    url='https://h5ctywhr.api.moji.com/weatherDetail'
    headers={'Content-Type': 'application/x-www-form-urlencoded; charset=UTF-8'}
    data={"cityId":cityId,"cityType":0}
    r=requests.post(url,headers=headers,json=data)
    result=r.json()

    # today forecast
    forecastToday=dict(
        humidity=f"{result['condition']['humidity']}%",
        temp=f"{result['condition']['temp']}°C",
        realFeel=f"{result['condition']['realFeel']}°C",
        weather=result['condition']['weather'],
        wind=f"{result['condition']['windDir']}{result['condition']['windLevel']}级",
        updateTime=(datetime.datetime.fromtimestamp(result['condition']['updateTime'])+datetime.timedelta(hours=8)).strftime('%H:%M:%S')
    )

    # 24 hours forecast
    forecastHours=[]
    for i in result['forecastHours']['forecastHour']:
        tmp={}
        tmp['PredictTime']=(datetime.datetime.fromtimestamp(i['predictTime'])+datetime.timedelta(hours=8)).strftime('%H:%M')
        tmp['Temperature']=i['temp']
        tmp['Body Temperature']=i['realFeel']
        tmp['Humidity']=i['humidity']
        tmp['Weather']=i['weather']
        tmp['Wind']=f"{i['windDesc']}{i['windLevel']}级"
        forecastHours.append(tmp)
    df_forecastHours=pd.DataFrame(forecastHours).set_index('PredictTime')

    # 7 days forecast
    forecastDays=[]
    day_format={1:'昨天',0:'今天',-1:'明天',-2:'后天'}
    for i in result['forecastDays']['forecastDay']:
        tmp={}
        now=datetime.datetime.fromtimestamp(i['predictDate'])+datetime.timedelta(hours=8)
        diff=(st.session_state.date_time-now).days
        festival=i['festival']
        tmp['PredictDate']=(day_format[diff] if diff in day_format else now.strftime('%m/%d')) + (f' {festival}' if festival != '' else '')
        tmp['Temperature']=f"{i['tempLow']}~{i['tempHigh']}°C"
        tmp['Humidity']=f"{i['humidity']}%"
        tmp['WeatherDay']=i['weatherDay']
        tmp['WeatherNight']=i['weatherNight']
        tmp['WindDay']=f"{i['windDirDay']}{i['windLevelDay']}级"
        tmp['WindNight']=f"{i['windDirNight']}{i['windLevelNight']}级"
        forecastDays.append(tmp)
    df_forecastDays=pd.DataFrame(forecastDays).set_index('PredictDate')
    return forecastToday,df_forecastHours,df_forecastDays

@st.experimental_singleton
def get_audio_bytes(music):
    audio_file = open(f'music/{music}-周杰伦.mp3', 'rb')
    audio_bytes = audio_file.read()
    audio_file.close()
    return audio_bytes

@st.experimental_singleton
def get_video_bytes():
    video_file = open(f'video/开不了口-广告曲.mp4', 'rb')
    video_bytes1 = video_file.read()
    video_file.close()
    video_file = open(f'video/最长的电影-广告曲.mp4', 'rb')
    video_bytes2 = video_file.read()
    video_file.close()
    return video_bytes1,video_bytes2

if __name__ == '__main__':
    main()