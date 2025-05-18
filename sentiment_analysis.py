import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from textblob import TextBlob  

# ======================
# Sentiment Analysis
# ======================

def analyze_sentiment(df):
    """Enhance sentiment analysis using TextBlob"""
    print("\n🔍 Analyzing sentiment...")
    
    # Calculate sentiment if not already present
    if 'sentiment' not in df.columns:
        df['sentiment'] = df['text'].apply(lambda x: TextBlob(str(x)).sentiment.polarity)

        
    if 'brand_mentioned' in df.columns:
        print("\n🔥 Top Brand Criticism")
        controversial_brands = ['H&M', 'Zara']  # Add other brands as needed
        for brand in controversial_brands:
            brand_criticism = df[df['brand_mentioned'] == brand].nsmallest(3, 'sentiment')
            if not brand_criticism.empty:
                print(f"\nMost negative {brand} posts:")
                print(brand_criticism[['text', 'sentiment', 'platform']].to_string(index=False))
            else:
                print(f"\nNo critical posts found for {brand}")
    

    
    # Categorize sentiment
    df['sentiment_category'] = pd.cut(df['sentiment'],
                                    bins=[-1, -0.5, -0.1, 0.1, 0.5, 1],
                                    labels=['Very Negative', 'Negative', 'Neutral', 'Positive', 'Very Positive'])
    
    return df

# ======================
# Visualization
# ======================

def generate_visualizations(df):
    """Create key visualizations"""
    print("\n📊 Generating visualizations...")
    
    plt.figure(figsize=(15, 10))
    
    # 1. Sentiment Distribution
    plt.subplot(2, 2, 1)
    sns.countplot(data=df, x='sentiment_category', palette='viridis')
    plt.title('Sentiment Distribution')
    plt.xticks(rotation=45)
    
    # 2. Platform Comparison
    plt.subplot(2, 2, 2)
    sns.boxplot(data=df, x='platform', y='sentiment', palette='Set2')
    plt.title('Sentiment by Platform')
    
    # 3. Brand Analysis
    if 'brand_mentioned' in df.columns:
        plt.subplot(2, 2, 3)
        top_brands = df['brand_mentioned'].value_counts().head(5).index
        sns.barplot(data=df[df['brand_mentioned'].isin(top_brands)],
                   x='brand_mentioned', y='sentiment', 
                   estimator='mean', palette='rocket')
        plt.title('Average Sentiment by Brand')
        plt.xticks(rotation=45)
    
    # 4. Word Cloud
    plt.subplot(2, 2, 4)
    text = ' '.join(df['text'].astype(str))
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title('Common Topics in Sustainable Fashion')
    
    plt.tight_layout()
    plt.savefig('sustainable_fashion_analysis.png')
    plt.show()

    # ===== ENGAGEMENT ANALYSIS =====
    plt.figure(figsize=(14, 8), facecolor='#f5f5f5')
    
    # Calculate engagement metrics
    engagement_metric = {
        'twitter': 'likes',
        'reddit': 'upvotes',
        'linkedin': 'reactions'
    }
    df['engagement'] = df.apply(
        lambda row: row[engagement_metric.get(row['platform'], 'likes')], 
        axis=1
    )
    
    # Calculate dynamic bounds to remove white space
    min_sentiment = df['sentiment'].min()
    max_sentiment = df['sentiment'].max()
    sentiment_padding = max(0.1, (max_sentiment - min_sentiment) * 0.05)  # 5% padding
    
    min_engagement = df['engagement'].min()
    max_engagement = df['engagement'].max()
    
    # Platform colors
    platform_colors = {
        'twitter': '#1DA1F2',
        'reddit': '#FF5700',
        'linkedin': '#ec87e4'
    }
    
    # Plot each platform
    for platform in ['twitter', 'reddit', 'linkedin']:
        platform_data = df[df['platform'] == platform]
        if not platform_data.empty:
            plt.scatter(
                x=platform_data['sentiment'],
                y=platform_data['engagement'],
                s=platform_data['text'].str.len() * 0.3,  # Scale dot size by text length
                c=platform_colors[platform],
                alpha=0.7,
                edgecolors='w',
                linewidth=0.5,
                label=platform.capitalize()
            )
    
    # Set optimized bounds
    plt.xlim(min_sentiment - sentiment_padding, max_sentiment + sentiment_padding)
    
    # Use symlog scale with optimized parameters
    plt.yscale('symlog', linthresh=1, linscale=0.5)
    
    # Calculate optimal y-axis limits
    if max_engagement > 0:
        y_lower = max(min_engagement * 0.9, 0.1)  # Don't go below 0.1
        y_upper = max_engagement * 1.1
        plt.ylim(y_lower, y_upper)
    
    # Styling
    plt.title("Sustainable Fashion Engagement Analysis\nHow Sentiment Drives Interactions", 
             fontsize=14, pad=20, fontweight='bold')
    plt.xlabel("Sentiment (Negative → Positive)", fontsize=12)
    plt.ylabel("Engagement (Log Scale)", fontsize=12)
    
    # Reference lines
    plt.axvline(0, color='gray', linestyle='--', alpha=0.5)
    
    # Legend
    legend = plt.legend(title='Platform', frameon=True, 
                      bbox_to_anchor=(1, 1), loc='upper left')
    legend.get_frame().set_linewidth(0.5)
    
    # Remove excess whitespace
    plt.tight_layout()
    
    plt.savefig('sustainable_fashion_engagement_optimized.png', 
               dpi=300, bbox_inches='tight', pad_inches=0.2)
    plt.show()

# ======================
# Trend Analysis
# ======================

def analyze_trends(df):
    """Identify temporal patterns"""
    print("\n⏳ Analyzing trends over time...")

    # Convert dates
    if not pd.api.types.is_datetime64_any_dtype(df['date']):
        df['date'] = pd.to_datetime(df['date'], errors='coerce')

    df = df.dropna(subset=['date'])  # Clean bad dates

    # Weekly trend
    df['week'] = df['date'].dt.to_period('W')
    weekly_trend = df.groupby('week')['sentiment'].mean().reset_index()
    weekly_trend['week'] = weekly_trend['week'].astype(str)

    # Plot weekly sentiment
    if not weekly_trend.empty:
        plt.figure(figsize=(12, 6))
        sns.lineplot(data=weekly_trend, x='week', y='sentiment', marker='o')
        plt.title('Weekly Sentiment Trend')
        plt.xticks(rotation=45)
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('weekly_sentiment_trend.png')
        plt.show()
        print("✅ Saved weekly sentiment trend: weekly_sentiment_trend.png")

   

    # ==== Event Impact Analysis  ====
    print("\n📅 Event Impact Analysis 2 (e.g., April Awareness Campaign)")
    event_date_2 = pd.to_datetime('2025-04-01')
    in_range_2 = df['date'].min() <= event_date_2 <= df['date'].max()

    if not in_range_2:
        print(f"⚠️ Event date {event_date_2.date()} is outside the dataset range: {df['date'].min().date()} to {df['date'].max().date()}")
        print("📉 Still generating dummy event sentiment plot...")

    df['pre_event_2'] = df['date'] < event_date_2
    event_impact_2 = df.groupby('pre_event_2')['sentiment'].agg(['mean', 'count'])
    event_impact_2.rename(index={True: 'Before Event', False: 'After Event'}, inplace=True)

    print(f"\nSentiment Change Around {event_date_2.date()}:")
    print(event_impact_2)

    plt.figure(figsize=(8, 4))
    sns.barplot(x=event_impact_2.index, y='mean', data=event_impact_2.reset_index())
    plt.title(f"Sentiment Before/After {event_date_2.date()}")
    plt.ylabel("Average Sentiment")
    plt.tight_layout()
    plt.savefig('event_sentiment_impact_2.png')
    plt.show()
    print("✅ Saved event sentiment impact plot: event_sentiment_impact_2.png")

    return weekly_trend


# ======================
# Main Execution
# ======================

if __name__ == "__main__":
    # Load your collected data
    try:
        df = pd.read_csv('sustainable_fashion_enriched_20250516_220045.csv')  # Replace with your actual filename
    except:
        print("⚠️ No data file found. Please run data collection first.")
        exit()
    
    # Run analysis pipeline
    df = analyze_sentiment(df)
    generate_visualizations(df)
    trend_data = analyze_trends(df)
    
    # Save enhanced data
    df.to_csv('sustainable_fashion_analyzed.csv', index=False)
    
    # Key insights report
    print("\n" + "="*60)
    print("📈 SUSTAINABLE FASHION INSIGHTS REPORT")
    print("="*60)
    print(f"\nTotal Posts Analyzed: {len(df)}")
    print(f"Overall Sentiment: {df['sentiment'].mean():.2f} (Scale: -1 to +1)")
    print("\nTop Positive Posts:")
    print(df.nlargest(3, 'sentiment')[['platform', 'text', 'sentiment']].to_string(index=False))
    print("\nTop Negative Posts:")
    print(df.nsmallest(3, 'sentiment')[['platform', 'text', 'sentiment']].to_string(index=False))
    print("\n" + "="*60)
    print("💾 Saved analyzed data to: sustainable_fashion_analyzed.csv")
    print("📊 Saved visualizations to: sustainable_fashion_analysis.png & weekly_sentiment_trend.png")
    print("📅 Saved event impact analysis to: event_sentiment_impact.png")
    print("="*60)
    print("\nThank you for using the Sustainable Fashion Analysis Tool! 🌱")