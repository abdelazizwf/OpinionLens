import random

from locust import HttpUser, between, task

text_pool = [
    "I loved this",
    "Bitch!",
    "Fugiat proident enim ad do dolore dolore ex id non consectetur reprehenderit consequat tempor. Esse officia velit non esse occaecat minim tempor labore. Aliqua sint sit nostrud quis officia est aute. Amet ut officia laborum ad sit pariatur.",
    "I went to buy milk like my dad did 20 years ago, I fell asleep at the mall and found him in my dreams playing catch with another kid. 10 out of 10 experience.",
    "Nien Nien Nien!",
    "My grandma is dead and I am sad",
    "renew visa softbank where",
    "I'd rather eat my own shoe than eat this shoe.",
    "this is me rn :-(",
]


class SimulatedUsers(HttpUser):
    wait_time = between(1, 5)

    @task
    def predict(self):
        text = random.choice(text_pool)
        url = "/api/v1/inference/predict"

        if random.random() > 0.5:
            url += f"?text={text}"
            self.client.get(url, name="/api/v1/inference/predict")
        else:
            self.client.post(url, json={"text": text})

    @task
    def batch_predict(self):
        k = random.randint(2, len(text_pool))
        batch = random.sample(text_pool, k=k)
        url = "/api/v1/inference/batch_predict"
        self.client.post(url, json=batch)
