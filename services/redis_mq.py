import redis
import threading
import time

class RedisMQ:
    def __init__(self, host="localhost", port="6379") -> None:
        self.host = host
        self.port = port
        self.client = redis.StrictRedis(host=host, port=port, db=0)

class RedisMQProducer(RedisMQ):
    def __init__(self, host="localhost", port="6379") -> None:
        super().__init__(host, port)
    
    def push(self, channel, message):
        self.client.publish(channel, message)

class RedisMQConsumer(RedisMQ):
    def __init__(self, host="localhost", port="6379", id_=None) -> None:
        super().__init__(host, port)
        self.id_ = id_
        self.subscribed_channel = {}
        self.is_alive = True
    
    def disable(self):
        self.is_alive = False
    
    def start(self):
        self.is_alive = True
    
    def subscribe_channel(self, channel, callback=None):
        if channel in self.subscribed_channel:
            return f"{self.id_} already subscribed {channel}"
        pubsub =  self.client.pubsub()
        pubsub.subscribe(channel)
        def listen(channel, pubsub):
            for item in pubsub.listen():
                if not self.is_alive:
                    break
                data = item[u'data'].decode('utf-8') if type(item[u'data']) == str else item[u'data']
                if data == b'exit':
                    self.unsubscribe_channel(channel)
                    break
                print(f"{channel} Received: {data}")
        # open a thread to listen to the new added channel
        listen_thread = threading.Thread(target=listen, args=[channel, pubsub])
        listen_thread.daemon = 2
        listen_thread.start()
        self.subscribed_channel[channel] = (pubsub, listen_thread)
        return f"{self.id_} start listen to {channel}"

    def close(self, channel):
        self.client.publish(channel=channel, message="exit", type= 'unsubscribe')

    def unsubscribe_channel(self, channel):
        pubsub, listen_thread = self.subscribed_channel[channel]
        pubsub.close()
        del self.subscribed_channel[channel]
        print(f"stop usubscribe channel {channel}")
        

producer = RedisMQProducer()
consumer_1 = RedisMQConsumer(id_="1")
print(consumer_1.subscribe_channel("a"))
print(consumer_1.subscribe_channel("b"))
print(consumer_1.subscribe_channel("b"))
producer.push("a", "I am a")
producer.push("b", "I am b")
producer.push("a", "I am a 2")
consumer_1.close("b")
producer.push("b", "I am b 2")
consumer_1.subscribe_channel("b")


while True:
    time.sleep(2)