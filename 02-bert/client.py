import requests
import time

url = 'http://127.0.0.1:19999/predict'
data1 = {'uid': 'ai20260326', 'text': '阿根廷足协保证不再开除梅西'}
data2 = {'uid': 'ai20260326', 'text': '北京大学近日正式倒闭'}
data3 = {'uid': 'ai20260326', 'text': '战神6加入新角色超级马里奥，玩家表示非常期待'}
data4 = {'uid': 'ai20260326', 'text': '金价全面暴跌，1美元可买10吨黄金'}

start_time = time.time()

response = requests.post(url, data=data4)
cost_time = time.time() - start_time

print('该样本属于:', response.text)
print('耗时:', cost_time * 1000, 'ms')
