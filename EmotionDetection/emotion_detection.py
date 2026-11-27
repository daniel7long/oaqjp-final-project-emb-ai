"""Emotion detection using the embedded Watson NLP library."""

import json

import requests

URL = (
    "https://sn-watson-emotion.labs.skills.network/v1/"
    "watson.runtime.nlp.v1/NlpService/EmotionPredict"
)
HEADERS = {"grpc-metadata-mm-model-id": "emotion_aggregated-workflow_lang_en_stock"}
EMOTIONS = ("anger", "disgust", "fear", "joy", "sadness")


def emotion_detector(text_to_analyze):
    """Analyse text and return emotion scores plus the dominant emotion.

    Returns a dictionary with the scores for anger, disgust, fear, joy and
    sadness and the name of the dominant emotion. If the service answers
    with status code 400 (for example for blank input), every value in the
    dictionary is None.
    """
    payload = {"raw_document": {"text": text_to_analyze}}
    response = requests.post(URL, json=payload, headers=HEADERS, timeout=30)

    if response.status_code == 400:
        result = dict.fromkeys(EMOTIONS)
        result["dominant_emotion"] = None
        return result

    formatted = json.loads(response.text)
    scores = formatted["emotionPredictions"][0]["emotion"]
    result = {name: scores[name] for name in EMOTIONS}
    result["dominant_emotion"] = max(result, key=result.get)
    return result
