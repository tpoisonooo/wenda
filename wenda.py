import torch
import threading
import os
import json
import datetime
from bottle import route, response, request, static_file, hook
import bottle
import argparse
import re

parser = argparse.ArgumentParser(description='Wenda config')
parser.add_argument('-c', type=str, dest="Config", default='config.xml', help="配置文件")
parser.add_argument('-p', type=int, dest="Port", help="使用端口号")
parser.add_argument('-l', type=bool, dest="Logging", help="是否开启日志")
parser.add_argument('-t', type=str, dest="LLM_Type", help="选择使用的大模型")
args = parser.parse_args()
print(args)
os.environ['wenda_'+'Config'] = args.Config 
os.environ['wenda_'+'Port'] = str(args.Port)
os.environ['wenda_'+'Logging'] = str(args.Logging)
os.environ['wenda_'+'LLM_Type'] = str(args.LLM_Type) 

from plugins.common import settings 
from plugins.common import error_helper 
from plugins.common import success_print 
from plugins.common import CounterLock


def load_LLM():
    try:
        from importlib import import_module
        LLM = import_module('plugins.llm_'+settings.LLM_Type)
        return LLM
    except Exception as e:
        print("LLM模型加载失败，请阅读说明：https://github.com/l15y/wenda", e)
LLM = load_LLM()

Logging=bool(settings.Logging == 'True')
if Logging:
    from plugins.defineSQL import session_maker, 记录

if not hasattr(LLM,"Lock") :
    mutex = CounterLock()
else:
    mutex = LLM.Lock()

@route('/static/<path:path>')
def staticjs(path='-'):
    if path.endswith(".html"):
        noCache()
    if path.endswith(".js"):
        return static_file(path, root="views/static/",mimetype ="application/javascript")
    return static_file(path, root="views/static/")

@route('/:name')
def static(name='-'):
    if name.endswith(".html"):
        noCache()
    return static_file(name, root="views")

from plugins.common import xml2json,json2xml
import json
@route('/api/readconfig', method=("POST","OPTIONS"))
def readconfig():
    allowCROS()
    with open(os.environ['wenda_'+'Config'],encoding = "utf-8") as f:
        j=xml2json(f.read(),True,1,1)
        # print(j)
        return json.dumps(j)
@route('/api/writeconfig', method=("POST","OPTIONS"))
def readconfig():
    allowCROS()
    data = request.json
    if not data:
        return '0'
    s=json2xml(data).decode("utf-8")
    with open(os.environ['wenda_'+'Config']+"_",'w',encoding = "utf-8") as f:
        f.write(s)
        # print(j)
        return s
# @route('/readxml')
# def readxml():
#     with open(os.environ['wenda_'+'Config'],encoding = "utf-8") as f:
#         return f.read()
@route('/api/llm')
def llm_js():
    noCache()
    return static_file('llm_'+settings.LLM_Type+".js", root="plugins")
    
@route('/api/plugins')
def read_auto_plugins():
    noCache()
    plugins=[]
    for root, dirs, files in os.walk("autos"):
        for file in files:
            if(file.endswith(".js")):
                file_path = os.path.join(root, file)
                with open(file_path, "r", encoding='utf-8') as f:
                    plugins.append( f.read())
    return "\n".join(plugins)
# @route('/writexml', method=("POST","OPTIONS"))
# def writexml():
    # data = request.json
    # s=json2xml(data).decode("utf-8")
    # with open(os.environ['wenda_'+'Config']+"_",'w',encoding = "utf-8") as f:
    #     f.write(s)
    #     # print(j)
    #     return s
def noCache():
    response.set_header("Pragma", "no-cache")
    response.add_header("Cache-Control", "must-revalidate")
    response.add_header("Cache-Control", "no-cache")
    response.add_header("Cache-Control", "no-store")
    
def allowCROS():
    response.set_header('Access-Control-Allow-Origin', '*')
    response.add_header('Access-Control-Allow-Methods', 'POST,OPTIONS')
    response.add_header('Access-Control-Allow-Headers',
                        'Origin, Accept, Content-Type, X-Requested-With, X-CSRF-Token')
@route('/')
def index():
    noCache()
    return static_file("index.html", root="views")



@route('/api/chat_now', method=('GET',"OPTIONS"))
def api_chat_now():
    allowCROS()
    noCache()
    return {'queue_length':mutex.get_waiting_threads()}


@hook('before_request')
def validate():
    REQUEST_METHOD = request.environ.get('REQUEST_METHOD')
    HTTP_ACCESS_CONTROL_REQUEST_METHOD = request.environ.get(
        'HTTP_ACCESS_CONTROL_REQUEST_METHOD')
    if REQUEST_METHOD == 'OPTIONS' and HTTP_ACCESS_CONTROL_REQUEST_METHOD:
        request.environ['REQUEST_METHOD'] = HTTP_ACCESS_CONTROL_REQUEST_METHOD


@route('/api/find', method=("POST","OPTIONS"))
def api_find():
    allowCROS()
    data = request.json
    if not data:
        return '0'
    prompt = data.get('prompt')
    step = data.get('step')
    if step is None:
        step = int(settings.library.Step)
    return json.dumps(zhishiku.find(prompt,int(step)))

@route('/chat/completions', method=("POST","OPTIONS"))
def api_chat_box():
    response.content_type = "text/event-stream"
    response.add_header("Connection", "keep-alive")
    response.add_header("Cache-Control", "no-cache")
    data = request.json
    max_length = data.get('max_tokens')
    if max_length is None:
        max_length = 2048
    top_p = data.get('top_p')
    if top_p is None:
        top_p = 0.2
    temperature = data.get('temperature')
    if temperature is None:
        temperature = 0.8
    use_zhishiku = data.get('zhishiku')
    if use_zhishiku is None:
        use_zhishiku = False
    messages = data.get('messages')
    prompt = messages[-1]['content']
    # print(messages)
    history_formatted = LLM.chat_init(messages)
    response_text = ''
    # print(request.environ)
    IP = request.environ.get(
        'HTTP_X_REAL_IP') or request.environ.get('REMOTE_ADDR')
    error = ""
    with mutex:
        yield "data: %s\n\n" %json.dumps({"response": (str(len(prompt))+'字正在计算')})

        print("\033[1;32m"+IP+":\033[1;31m"+prompt+"\033[1;37m")
        try:
            for response_text in LLM.chat_one(prompt, history_formatted, max_length, top_p, temperature, zhishiku=use_zhishiku):
                if (response_text):
                    # yield "data: %s\n\n" %response_text
                    yield "data: %s\n\n" %json.dumps({"response": response_text})
            
            yield "data: %s\n\n" %"[DONE]"
        except Exception as e:
            error = str(e)
            print("错误", settings.red, error, settings.white)
            response_text = ''
        torch.cuda.empty_cache()
    if response_text == '':
        yield "data: %s\n\n" %json.dumps({"response": ("发生错误，正在重新加载模型"+error)})
        os._exit(0)


def search_bing(search_query,step = 0):
    import requests
    import re
    search_query = search_query.strip()
    url = 'http://www.bing.com/search?q={}'.format(search_query)
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/94.0.4606.61 Safari/537.36 Edg/94.0.992.31'}
    
    import pdb
    pdb.set_trace()
    with requests.Session() as session:
        res = session.get(url, headers=headers)

    from bs4 import BeautifulSoup as BS
    soup = BS(res.text, 'html.parser')
    td = soup.findAll("h2")
    items = []
    for t in td:
        a = t.find_all('a')[0]

        href = a.get('href')
        text = a.text
        item = '* [{}]({})'.format(text, href)
        items.append(item)

        if len(items) >= 3:
            break
    
    ret = ''
    if len(items) > 0:
        ret = '\n\n更多：\n' + '\n'.join(items)
    return ret


@route('/api/chat_stream', method=("POST","OPTIONS"))
def api_chat_stream():
    allowCROS()
    data = request.json
    if not data:
        return '0'
    prompt = data.get('prompt')
    max_length = data.get('max_length')
    if max_length is None:
        max_length = 2048
    top_p = data.get('top_p')
    if top_p is None:
        top_p = 0.7
    temperature = data.get('temperature')
    if temperature is None:
        temperature = 0.9
    # use_zhishiku = data.get('zhishiku')
    # if use_zhishiku is None:
    #     use_zhishiku = False
    use_zhishiku = True
    keyword = data.get('keyword')
    if keyword is None:
        keyword = prompt
    history = data.get('history')
    history_formatted = LLM.chat_init(history)
    response = ''
    # print(request.environ)
    IP = request.environ.get(
        'HTTP_X_REAL_IP') or request.environ.get('REMOTE_ADDR')
    error = ""
    footer = '///'

    bing = search_bing(prompt)
    if use_zhishiku:
        # print(keyword)
        response_d = zhishiku.find(keyword,int(settings.library.Step))
        output_sources = [i['title'] for i in response_d]
        results = '\n'.join([str(i+1)+". "+re.sub('\n\n', '\n', response_d[i]['content']) for i in range(len(response_d))])
        prompt = 'system: 请扮演一名程序员，根据以下内容回答问题：'+prompt + "\n"+ results
        # if bool(settings.library.Show_Soucre == 'True'):
        #     footer = "\n### 来源：\n"+('\n').join(output_sources)+'///'
    
    if len(output_sources) > 0:

        with mutex:
            print("\033[1;32m"+IP+":\033[1;31m"+prompt+"\033[1;37m")
            try:
                val = None
                for response in LLM.chat_one(prompt, history_formatted, max_length, top_p, temperature, zhishiku=use_zhishiku):
                    if (response):
                        value = response
                        # yield response+footer
                if bool(settings.library.Show_Soucre == 'True'):
                    # value = value + "\n###来源：\n"+('\n').join(output_sources)
                    value = value + "\n\n参考：\n"+('\n').join(output_sources)

                value = value + bing
                yield value + footer
            except Exception as e:
                error = str(e)
                print("错误", settings.red, error, settings.white)
                response = ''
                # raise e
            torch.cuda.empty_cache()
        if response == '':
            yield "发生错误，正在重新加载模型"+error+'///'
            os._exit(0)
        if Logging:
            with session_maker() as session:
                jl = 记录(时间=datetime.datetime.now(), IP=IP, 问=prompt, 答=response)
                session.add(jl)
                session.commit()
        print(response)
    else:
        yield "知识库没结果" + footer
    yield "/././"


model = None
tokenizer = None


def load_model():
    with mutex:
        LLM.load_model()
    torch.cuda.empty_cache()
    success_print("模型加载完成")


thread_load_model = threading.Thread(target=load_model)
thread_load_model.start()
zhishiku = None

def load_zsk():
    try:
        from importlib import import_module
        global zhishiku
        zhishiku = import_module('plugins.zhishiku_'+settings.library.Type)
        success_print("知识库加载完成")
    except Exception as e:
        error_helper("知识库加载失败，请阅读说明",r"https://github.com/l15y/wenda#%E7%9F%A5%E8%AF%86%E5%BA%93")
        raise e

thread_load_zsk = threading.Thread(target=load_zsk)
thread_load_zsk.start()
bottle.debug(True)

# import webbrowser
# webbrowser.open_new('http://127.0.0.1:'+str(settings.Port))

import functools
def pathinfo_adjust_wrapper(func):
    # A wrapper for _handle() method
    @functools.wraps(func)
    def _(s,environ):
        environ["PATH_INFO"] = environ["PATH_INFO"].encode("utf8").decode("latin1")
        return func(s,environ)
    return _
bottle.Bottle._handle = pathinfo_adjust_wrapper(bottle.Bottle._handle)#修复bottle在处理utf8 url时的bug

bottle.run(server='paste', host="0.0.0.0", port=settings.Port, quiet=True)
