import os
import urllib
import tarfile

def download_model(source_url,  target_file):
    global downloaded
    downloaded = 0
    def show_progress(count, block_size, total_size):
        global downloaded
        downloaded += block_size
        print('downloading ... %d%%' % round(((downloaded*100.0) / total_size)))

    print('downloading ... ')
    urllib.urlretrieve(source_url, filename=target_file, reporthook=show_progress)
    print('downloading ... done')

    print('extracting ...')
    tar = tarfile.open(target_file, "r:gz")
    tar.extractall()
    tar.close()
    os.remove(target_file)
    print('extracting ... done')


if __name__ == '__main__':
    source_url = 'https://nuage.lix.polytechnique.fr/index.php/s/BqiX5rcWszkKT9N/download'
    target_dir = os.path.dirname(os.path.abspath(__file__))
    target_file = os.path.join(target_dir, 'demo_dfaust_data.tar.gz')
    download_model(source_url,   target_file)
