from app import app
from extensions import mongo

app.app_context().push()
mongo.db.app_settings.update_one({'_id': 'app_settings'}, {'$set': {'maintenance_mode': True}}, upsert=True)

client = app.test_client()
response = client.get('/this-route-does-not-exist')
print("Status:", response.status_code)
print(response.data.decode('utf-8')[:200])
