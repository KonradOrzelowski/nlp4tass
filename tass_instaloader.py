import datetime
from instaloader import Instaloader, Profile, TopSearchResults

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
                k += 1
                if k == k_max:
                    break
                else:
                    continue
            else:
                self.L.download_post(post, target = profile.username)
                k = 0  # set k to 0


    def get_posts_for_profile(self, profile: Profile, since: datetime, until: datetime) -> None:
        posts = profile.get_posts()
        self.download_post_from_time_range(profile, posts, since, until)

def main():
    # initiate TassInstaloader
    tass_instaloader = TassInstaloader('your_username', 'your_password')
    
    # get profile
    person_name = 'nlp tass'
    profile = tass_instaloader.get_profile(person_name)
    
    # download posts from 2019-01-01 to 2019-01-31
    tass_instaloader.get_posts_for_profile(profile, since = datetime(2019, 1, 1),
                                                         until = datetime(2019, 1, 31))

# if __name__ == '__main__':
#     main()
