from urllib.request import urlopen
import json, csv, re

robot_emotes = [":)", ":(", ":o", ":z", "B)", ":\\", ":|", ";)", ";p", ":p", ":>", "<]", ":7", "R)", "o_O", "#/", ":D", ">(", "<3", "lul", "lol"]

pos_emo = ['PogChamp', '4Head', 'EleGiggle', 'Kappa', ":)", ":o", "B)", ";)", ";p", ":p", ":>", "<]", ":D", "<3", "MingLee", "Kreygasm", "TakeNRG", "GivePLZ", "HeyGuys", "SeemsGood", "VoteYea", "Poooound", "AMPTropPunch", "CoolStoryBob", "BloodTrail", "FutureMan", "FunRun", "VoHiYo", "gg"]
neg_emo = [">(", ":(", ":\\", ":z", 'WutFace', "BabyRage", "FailFish", "DansGame", "BibleThump", "NotLikeThis", "PJSalt", "SwiftRage", "ResidentSleeper", "VoteNay", "BrokeBack", "rage"]
sentiment = 0


def check_sentiment(emote):
	for e in pos_emo:
		if emote.find(e.lower()) >= 0:
			return 1
	for e in neg_emo:
		if emote.find(e.lower()) >= 0:
			return -1
	return 0


# Retrieve data from sub emotes
url = "https://twitchemotes.com/api_cache/v2/subscriber.json"
response = urlopen(url)
data = response.read().decode("utf-8")
data = json.loads(data)

with open('sub_emotes.csv', 'w') as csvfile:
	field_names = ['emotes', 'sentiment']
	writer = csv.DictWriter(csvfile, fieldnames=field_names)
	writer.writeheader()
	for key in data['channels'].keys():
		for c in data['channels'][key]['emotes']:
			sentiment = check_sentiment(c['code'].lower())
			writer.writerow({'emotes': c['code'].lower(),
			                 'sentiment': sentiment})

	for c in data['unknown_emotes']['emotes']:
		sentiment = check_sentiment(c['code'].lower())
		writer.writerow({'emotes': c['code'].lower(),
		                 'sentiment': sentiment})


# Retrieve data from global emotes
url = 'https://twitchemotes.com/api_cache/v2/global.json'
response = urlopen(url)
data = response.read().decode("utf-8")
data = json.loads(data)

with open('global_emotes.csv', 'w') as csvfile:
	field_names = ['emotes', 'sentiment']
	writer = csv.DictWriter(csvfile, fieldnames=field_names)
	writer.writeheader()
	for key in data['emotes']:
		sentiment = check_sentiment(key.lower())
		writer.writerow({'emotes': key.lower(), 
		                 'sentiment': sentiment})

	# write robot_emotes to global
	for robot in robot_emotes:
		sentiment = check_sentiment(robot.lower())
		writer.writerow({'emotes': robot,
		                 'sentiment': sentiment})


