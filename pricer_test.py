import modal

app = modal.App("pricer-test")

@app.local_entrypoint()
def main():
    text = "Quadcast HyperX condenser mic, connects via usb-c to your computer for crystal clear audio"
    Pricer = modal.Cls.from_name("pricer-service", "Pricer")
    pricer = Pricer()
    reply = pricer.price.remote(text)
    print(reply)