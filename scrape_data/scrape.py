import praw
from praw.models import MoreComments
from psaw import PushshiftAPI
import datetime as dt
import json
import time
import os

current = time.time()
def measure():
    global current
    t = time.time()
    print(t - current)
    current = t


def writeDataFor(subredditName, postsPerMonth, postScoreThreshold, commentScoreThreshold):
    reddit = praw.Reddit(
        client_id = "tXoEIYnD1_R8TcjIidZT6Q",
        client_secret = "RwBS56RbpHfRnIqr4ocvJJmfu3zGyw",
        username = "ESOGlokta",
        password = os.getenv('REDDIT_PASSWORD'),
        user_agent = "my user agent",
    )
    psaw_api = PushshiftAPI(reddit)

    data = []

    for y in range(2015, 2023):
        for m in range(1, 13):

            measure()
            print(y, m)

            start_epoch = int(dt.datetime(y, m, 1).timestamp())
            end_epoch = int(dt.datetime(y+1 if m == 12 else y, 1 if m == 12 else m+1, 1).timestamp())

            gen = psaw_api.search_submissions(
                after = start_epoch,
                before = end_epoch,
                subreddit = subredditName,
                filter = ['id'],
                limit = postsPerMonth
            )

            for submission in gen:
                postId = submission.id
                post = reddit.submission(id=postId)
                if (post.score >= postScoreThreshold):
                    title = post.title
                    comments = []
                    post.comments.replace_more(limit=0)
                    for comment in post.comments:
                        if (comment.score >= commentScoreThreshold):
                            comments.append(comment.body)
                    if (len(comments) > 0):
                        data.append({"title": title, "comments": comments})

    # Serializing json
    json_object = json.dumps(data, indent=4)
    
    # Writing to file.json
    with open(subredditName+".json", "w") as outfile:
        outfile.write(json_object)

# The limiting factor for runtime is the number of posts observed
# Reddit only allows a logged-in account to access 1 post per second
# With 100 posts per month, will view ~10,000 posts total in ~3 hours
# Low thresholds set in order to get more data per post observed
# writeDataFor("FreeCompliments", 100, 4, 2)
writeDataFor("RoastMe", 100, 10, 4)

