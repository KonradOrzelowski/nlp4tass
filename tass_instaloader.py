import datetime
from instaloader import Instaloader, Profile

class TassInstaloader:
    
    def __init__(self, user: str, password: str):
        self.user = user
        self.password = password
        
        # Login
        self.L = Instaloader()
        self.L.login(self.user, self.password)
    
    def get_profile(self, profile_name: str):
        return Profile.from_username(self.L.context, profile_name)
    
    def download_post_from_time_range(self, profile: Profile, posts : list,\
                                      since: datetime, until: datetime, k_max: int = 50):
        k = 0  # initiate k
        
        for post in posts:
            postdate = post.date
        
            if postdate > until:
                continue
            elif postdate <= since:
                k += 1
                if k == k_max:
                    break
                else:
                    continue
            else:
                self.L.download_post(post, target = profile.username)
                k = 0  # set k to 0

if __name__ == '__main__':
    # initiate TassInstaloader
    tass_instaloader = TassInstaloader('your_username', 'your_password')
    
    # get profile
    profile = tass_instaloader.get_profile('tass')
    
    # get posts
    posts = profile.get_posts()
    
    # download posts from 2019-01-01 to 2019-01-31
    tass_instaloader.download_post_from_time_range(profile, posts, since = datetime(2019, 1, 1), until = datetime(2019, 1, 31))