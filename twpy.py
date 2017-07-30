import tweepy
import json

with open("./twitter_auth.json") as f:
    twitter_auth = json.load(f)

auth = tweepy.OAuthHandler(twitter_auth["CONSUMER_KEY"], twitter_auth["CONSUMER_SECRET"])

auth.set_access_token(twitter_auth["ACCESS_TOKEN"], twitter_auth["ACCESS_SECRET"])

api = tweepy.API(auth)
