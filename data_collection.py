import pandas as pd
from datetime import datetime, timedelta
import time
import random
from collections import Counter
import requests

# ======================
# Enhanced Data Collection
# ======================

def get_enriched_twitter_data():
    """Generate Twitter data with enhanced metadata"""
    base_tweets = [
        {"text": "Thrift shopping is my go-to for sustainable fashion. Reduce, reuse, recycle! ♻️ #SlowFashion",
         "hashtags": ["SlowFashion"], "mentions": [], "lang": "en"},
        {"text": "Disappointed with @HnM's new 'conscious' line. This is pure greenwashing!",
         "hashtags": [], "mentions": ["HnM"], "lang": "en"},
        {"text": "Just visited @Reformation's new store. Their transparent supply chain sets the standard! #EthicalFashion",
         "hashtags": ["EthicalFashion"], "mentions": ["Reformation"], "lang": "en"}
    ]
    
    enriched_data = []
    for tweet in base_tweets * 5:
        full_text = tweet["text"]
        enriched_data.append({
            "platform": "twitter",
            "date": datetime.now() - timedelta(days=random.randint(0, 30)),
            "text": full_text,
            "likes": random.randint(10, 5000),
            "retweets": random.randint(2, 200),
            "replies": random.randint(0, 50),
            "hashtags": ", ".join(tweet["hashtags"]),
            "mentions": ", ".join(tweet["mentions"]),
            "language": tweet["lang"],
            "url": f"https://twitter.com/user/status/{random.randint(10000,99999)}",
            "user_type": random.choice(["consumer", "influencer", "brand"]),
            "media_attached": random.choice([True, False]),
            "sentiment": random.uniform(-0.5, 0.9),
            "word_count": len(full_text.split())
        })
    return enriched_data

def get_enriched_reddit_data():
    """Generate Reddit data with thread context"""
    base_posts = [
        {"title": "How to identify greenwashing in fashion brands?", "body": "Looking for reliable certification standards..."},
        {"title": "My 6-month capsule wardrobe experiment", "body": "Reduced my clothing purchases by 80%!"},
        {"title": "Fast fashion's environmental impact visualized", "body": "Shocking data from new research study"}
    ]
    
    subreddits = ["SustainableFashion", "EthicalFashion", "ZeroWaste", "AntiConsumption"]
    enriched_data = []
    
    for post in base_posts * 5:
        sub = random.choice(subreddits)
        full_text = f"{post['title']} {post['body']}"
        word_count = len(full_text.split())
        
        enriched_data.append({
            "platform": "reddit",
            "subreddit": sub,
            "date": datetime.now() - timedelta(days=random.randint(0, 60)),
            "title": post["title"],
            "text": post["body"],
            "full_text": full_text,
            "upvotes": random.randint(10, 2500),
            "comments": random.randint(3, 150),
            "awards": random.randint(0, 5),
            "flair": random.choice(["Discussion", "Article", "Personal", None]),
            "url": f"https://reddit.com/r/{sub}/comments/{random.randint(10000,99999)}",
            "word_count": word_count,
            "reading_time": round(word_count/200, 1),
            "sentiment": random.uniform(-0.3, 0.7)
        })
    return enriched_data

def get_linkedin_data():
    """Simulate LinkedIn professional perspectives"""
    posts = []
    companies = ["Patagonia", "Stella McCartney", "Eileen Fisher", "Allbirds", "Veja"]
    
    for _ in range(10):
        post_type = random.choice(["article", "post", "news"])
        company = random.choice(companies)
        post_text = f"{company}'s new circular fashion initiative: {random.choice(['recycling program', 'rental service', 'repair workshops'])}"
        
        posts.append({
            "platform": "linkedin",
            "date": datetime.now() - timedelta(days=random.randint(0, 90)),
            "author": f"{company} Sustainability Team",
            "text": post_text,
            "reactions": random.randint(50, 5000),
            "comments": random.randint(5, 200),
            "post_type": post_type,
            "url": f"https://linkedin.com/feed/update/{random.randint(1000000,9999999)}",
            "sentiment": random.uniform(0.1, 0.8),
            "word_count": len(post_text.split())
        })
    return posts

def enrich_with_external_data(df):
    """Add external datasets"""
    brand_scores = {
        "Patagonia": 0.9, "HnM": 0.2, "Reformation": 0.7,
        "Zara": -0.1, "Stella McCartney": 0.8
    }
    
    # Extract brand mentions from both @mentions and text
    df['brand_mentioned'] = df['text'].str.extract(r'(?:@|^| )(HnM|Reformation|Patagonia|Zara|Stella McCartney)')[0]
    
    # Add brand sentiment if mentioned
    df['brand_sentiment'] = df['brand_mentioned'].map(brand_scores)
    
    # Add post characteristics
    if 'word_count' not in df.columns:
        df['word_count'] = df['text'].str.split().str.len()
    df['has_hashtags'] = df['text'].str.contains('#')
    df['is_negative'] = df['sentiment'] < 0
    df['engagement_rate'] = df.apply(lambda x: 
        (x.get('likes', 0) + x.get('upvotes', 0) + x.get('reactions', 0)) / 
        max(x['word_count'], 1), axis=1)
    
    return df

# ======================
# Analysis & Export
# ======================

if __name__ == "__main__":
    print("🚀 Collecting enriched sustainable fashion data...")
    
    try:
        # Get data from all sources
        twitter_data = get_enriched_twitter_data()
        reddit_data = get_enriched_reddit_data()
        linkedin_data = get_linkedin_data()
        
        # Combine and enrich
        df = pd.DataFrame(twitter_data + reddit_data + linkedin_data)
        df = enrich_with_external_data(df)
        
        # Clean data
        df.fillna({
            'brand_mentioned': 'Unknown',
            'brand_sentiment': 0,
            'flair': 'None',
            'subreddit': 'Other'
        }, inplace=True)
        
        # Save with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"sustainable_fashion_enriched_{timestamp}.csv"
        df.to_csv(filename, index=False)
        
        # Analysis
        print(f"\n✅ Collected {len(df)} enriched posts")
        print("Platform distribution:")
        print(df['platform'].value_counts())
        
        print("\n📊 Key metrics:")
        print(f"Average sentiment: {df['sentiment'].mean():.2f}")
        print(f"Most active subreddit: {df[df['platform'] == 'reddit']['subreddit'].mode()[0]}")
        print(f"Top mentioned brands:\n{df['brand_mentioned'].value_counts().head(5)}")
        print(f"\n💾 Saved to {filename}")
        print(f"Columns collected: {list(df.columns)}")
        
    except Exception as e:
        print(f"\n❌ Error occurred: {str(e)}")
        print("Please check your inputs and try again.")