import datetime
from instaloader import Instaloader, Profile, TopSearchResults



import time as time
import random as random

class TassInstaloader:
    
    def __init__(self, user: str, password: str):
        self.user = user
        self.password = password
        
        # Login
        self.L = Instaloader()
        self.L.login(self.user, self.password)
    
    def get_profile(self, profile_name: str) -> Profile:
        '''Get profile by profile name'''
        return Profile.from_username(self.L.context, profile_name)
    
    def find_profile_by_name(self, person_name: str) -> Profile:
        '''Find profile by person name'''
        # Get the top search results for the given query
        search_results = TopSearchResults(self.context, person_name)
        # Get users and number of followers
        users = {username.username: username.followers for username in search_results.get_profiles()}

        # Sort the dictionary by number of followers
        sorted_users = sorted(users.items(), key=lambda x: x[1], reverse=True)

        # Get profile for the top user
        return Profile.from_username(self.L.context, sorted_users[0][0])

    def download_post_from_time_range(self, profile: Profile, posts : list,\
                                      since: datetime, until: datetime, k_max: int = 50):
        '''Download posts from time range profile'''
        k = 0  # initiate k
        
        for post in posts:
            postdate = post.date
        
            if postdate > until:
                continue
            elif postdate <= since:
                # k += 1
                if k == k_max:
                    break
                else:
                    continue
            else:
                k += 1
                time.sleep(random.randint(10,15))
                self.L.download_post(post, target = profile.username)
                # k = 0  # set k to 0
                if k == k_max:
                    break
    def download_X_popular(self, profile, X):
        posts = profile.get_posts()
        posts_sorted_by_likes = sorted(posts,
                               key=lambda p: p.likes + p.comments,
                               reverse=True)
        
        for post in posts_sorted_by_likes[0:X]:
            time.sleep(random.randint(9, 18))
            self.L.download_post(post, target = profile.username)


    def get_posts_for_profile(self, profile: Profile, since: datetime, until: datetime) -> None:
        posts = profile.get_posts()
        self.download_post_from_time_range(profile, posts, since, until)

def main():
    tass_instaloader = TassInstaloader('user', 'password')
    
    profiles = ['harrykane', 'trentarnold66', 'sterling7', 'kylewalker2',
                'philfoden', 'reecejames', 'jackgrealish', 'ktrippier2',
                'declanrice', 'madders', 'bukayosaka87', 'masonmount']
    
    
    for person_name in profiles:
        first, second = random.randrange(9, 18), random.randrange(9, 18)
        print(f"{person_name} first wait {first}")
        
        profile = tass_instaloader.get_profile(person_name)
        time.sleep(first)
        posts = profile.get_posts()
    
        
        # download posts from 2019-01-01 to 2019-01-31
        tass_instaloader.download_post_from_time_range(profile, posts,
                                                        datetime.datetime(2020, 11, 1),
                                                        datetime.datetime(2023, 1, 1), k_max = 15)
        print(f"{person_name} second {second}")
        # tass_instaloader.download_X_popular(profile, 20)
        
        time.sleep(second)





