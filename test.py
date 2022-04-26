import argparse
import json

import cv2
import numpy as np
import requests


def softmax(x):
    x = np.array(x)
    res = np.round(np.exp(x) / sum(np.exp(x)) * 100, 2)
    return {k: v for k, v in enumerate(res)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=".")
    parser.add_argument(
        "--filename", metavar="N", type=str, help="an integer for the accumulator"
    )
    endpoint = "http://localhost:5001/invocations"
    headers = {"Content-Type": "application/json"}
    img = cv2.imread(parser.parse_args().filename)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) / 255.0
    img = img[np.newaxis, np.newaxis, :, :]
    data = json.dumps({"inputs": img.tolist()})
    r = requests.post(endpoint, data=data, headers=headers)
    res = softmax(r.json()[0])
    for i, v in res.items():
        print(f"probability to be {i}: {v}%")
