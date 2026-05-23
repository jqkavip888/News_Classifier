import requests
import time

url = 'http://127.0.0.1:19999/predict'
data = {'uid':'ai20260326','text':'阿根廷足协保证不再开除梅西'}

start_time = time.time()
response = requests.post(url,data=data)
end_time = time.time()
print(response.text)
print(f'cost time: {end_time - start_time} s')