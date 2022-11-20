import praw
from praw.models import MoreComments
from psaw import PushshiftAPI
import datetime as dt
import json

def writeDataFor(subredditName, postScoreThreshold, commentScoreThreshold):
    print("Start")
    r = praw.Reddit(
        client_id="tXoEIYnD1_R8TcjIidZT6Q",
        client_secret="RwBS56RbpHfRnIqr4ocvJJmfu3zGyw",
        user_agent="my user agent",
    )
    api = PushshiftAPI(r)

    data = []

    start_epoch=int(dt.datetime(2022, 11, 1).timestamp())

    print("search")
    gen = api.search_submissions(
        after = start_epoch,
        subreddit = subredditName,
        filter = ['title', 'comments'],
        limit = 100
    )
                        
    results = list(gen)

    for post in results:
        if (post.score >= postScoreThreshold):
            print(len(post.comments))
            comments = []
            for comment in post.comments:
                if (not isinstance(comment, MoreComments) and comment.score >= commentScoreThreshold):
                    comments.append(comment.body)
            if (len(comments) > 0):
                data.append({"title": post.title, "comments": comments})

    # Serializing json
    json_object = json.dumps(data, indent=4)
    
    # Writing to file.json
    with open(subredditName+".json", "w") as outfile:
        outfile.write(json_object)


writeDataFor("FreeCompliments", 10, 3)

