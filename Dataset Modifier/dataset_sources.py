# the file that contains all csv files, their resource, and their type(fake, real, both)

# Fake:0 Real:1

# all csv files to be combined
datasets = [
    'news_articles.csv',
    'WELFake_Dataset.csv',
    'True-clmentbisaillon.csv',
    'Fake-clmentbisaillon.csv',
    'PolitiFact_fake_news_content.csv',
    'PolitiFact_real_news_content.csv',
    'BuzzFeed_fake_news_content.csv',
    'BuzzFeed_real_news_content.csv',
    'argilla.csv',
    'fakenewsenglish_combined.csv',
    'test-erfan.csv',
    'train-erfan.csv',
    'validation-erfan.csv'
]

# URLs that the datasets are obtained from
dataset_urls = {
    'news_articles.csv': 'https://www.kaggle.com/datasets/ruchi798/source-based-news-classification/',
    'WELFake_Dataset.csv': 'https://www.kaggle.com/datasets/saurabhshahane/fake-news-classification',
    'True-clmentbisaillon.csv': 'https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset',
    'Fake-clmentbisaillon.csv': 'https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset',
    'PolitiFact_fake_news_content.csv': 'https://www.kaggle.com/datasets/mdepak/fakenewsnet',
    'PolitiFact_real_news_content.csv': 'https://www.kaggle.com/datasets/mdepak/fakenewsnet',
    'BuzzFeed_fake_news_content.csv': 'https://www.kaggle.com/datasets/mdepak/fakenewsnet',
    'BuzzFeed_real_news_content.csv': 'https://www.kaggle.com/datasets/mdepak/fakenewsnet',
    'argilla.csv': 'https://huggingface.co/datasets/argilla/news-fakenews',
    'fakenewsenglish_combined.csv': 'https://huggingface.co/datasets/pushpdeep/fake_news_combined',
    'test-erfan.csv': 'https://huggingface.co/datasets/ErfanMoosaviMonazzah/fake-news-detection-dataset-English',
    'train-erfan.csv': 'https://huggingface.co/datasets/ErfanMoosaviMonazzah/fake-news-detection-dataset-English',
    'validation-erfan.csv': 'https://huggingface.co/datasets/ErfanMoosaviMonazzah/fake-news-detection-dataset-English'

}

dataset_fake_or_real = {
    'news_articles.csv': 'both',
    'WELFake_Dataset.csv': 'both',
    'True-clmentbisaillon.csv': 'real',
    'Fake-clmentbisaillon.csv': 'fake',
    'PolitiFact_fake_news_content.csv': 'fake',
    'PolitiFact_real_news_content.csv': 'real',
    'BuzzFeed_fake_news_content.csv': 'fake',
    'BuzzFeed_real_news_content.csv': 'real',
    'argilla.csv': 'both',
    'fakenewsenglish_combined.csv': 'both',
    'test-erfan.csv': 'both',
    'train-erfan.csv': 'both',
    'validation-erfan.csv': 'both'
}