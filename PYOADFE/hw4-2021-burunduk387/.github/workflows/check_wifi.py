#!/usr/bin/env python3

import json

messages = [
    "Hello, World!",
    "Varo Haeldir",
    "xUUUx",
    "Both agreed yet hunted journey real servants ask devonshire principle park sixteen hunted.",
]

for i, m in enumerate(messages):
    with open("wifi_{}.json".format(i)) as f:
        data = json.load(f)

        assert(data['message'], m)
