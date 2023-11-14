

datasets = [
    'news_articles',
    'fake-mrisdal',
    'Fake-jainpooja',
    'True-jainpooja',
    'WELFake_Dataset',
    'True-clmentbisaillon',
    'Fake-clmentbisaillon',
    'PolitiFact_fake_news_content',
    'PolitiFact_real_news_content',
    'BuzzFeed_fake_news_content',
    'BuzzFeed_real_news_content',
    'True-ISOT',
    'Fake-ISOT',
    'argilla',
    'fakenewsenglish_combined',
    'test-erfan'
    'train-erfan',
    'validation-erfan',
    'Fake-kunalsharma',
    'True-kunalsharma'


]


dataset_urls = {
    'news_articles': 'https://www.kaggle.com/datasets/ruchi798/source-based-news-classification/',
    'fake-mrisdal': 'https://www.kaggle.com/datasets/mrisdal/fake-news',
    'Fake-jainpooja': 'https://www.kaggle.com/datasets/jainpooja/fake-news-detection',
    'True-jainpooja': 'https://www.kaggle.com/datasets/jainpooja/fake-news-detection',
    'WELFake_Dataset': 'https://www.kaggle.com/datasets/saurabhshahane/fake-news-classification',
    'True-clmentbisaillon':' https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset',
    'Fake-clmentbisaillon': 'https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset',
    'PolitiFact_fake_news_content': 'https://www.kaggle.com/datasets/mdepak/fakenewsnet',
    'PolitiFact_real_news_content': 'https://www.kaggle.com/datasets/mdepak/fakenewsnet',
    'BuzzFeed_fake_news_content': 'https://www.kaggle.com/datasets/mdepak/fakenewsnet',
    'BuzzFeed_real_news_content': 'https://www.kaggle.com/datasets/mdepak/fakenewsnet',
    'True-ISOT': 'https://www.kaggle.com/datasets/csmalarkodi/isot-fake-news-dataset',
    'Fake-ISOT': 'https://www.kaggle.com/datasets/csmalarkodi/isot-fake-news-dataset',
    'argilla': 'https://huggingface.co/datasets/argilla/news-fakenews',
    'fakenewsenglish_combined': 'https://huggingface.co/datasets/pushpdeep/fake_news_combined',
    'test-erfan': 'https://huggingface.co/datasets/ErfanMoosaviMonazzah/fake-news-detection-dataset-English',
    'train-erfan': 'https://huggingface.co/datasets/ErfanMoosaviMonazzah/fake-news-detection-dataset-English',
    'validation-erfan': 'https://huggingface.co/datasets/ErfanMoosaviMonazzah/fake-news-detection-dataset-English',
    'Fake-kunalsharma': 'https://huggingface.co/datasets/kunalsharma/fake-news',
    'True-kunalsharma': 'https://huggingface.co/datasets/kunalsharma/fake-news'



}

dataset_fake_or_real = {
    'news_articles': 'both',
    'fake': 'fake',
    'Fake-jainpooja': 'fake',
    'True-jainpooja': 'real',
    'WELFake_Dataset': 'both',
    'True-clmentbisaillon': 'real',
    'Fake-clmentbisaillon': 'fake',
    'PolitiFact_fake_news_content': 'fake',
    'PolitiFact_real_news_content': 'real',
    'BuzzFeed_fake_news_content': 'fake',
    'BuzzFeed_real_news_content': 'real',
    'True-ISOT': 'real',
    'Fake-ISOT': 'fake',
    'argilla': 'both',
    'fakenewsenglish_combined': 'both',
    'test-erfan': 'both',
    'train-erfan': 'both',
    'validation-erfan': 'both',
    'Fake-kunalsharma': 'fake',
    'True-kunalsharma': 'real'

}