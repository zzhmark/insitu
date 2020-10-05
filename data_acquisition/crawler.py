import os
from time import sleep
from urllib.request import urlretrieve

from selenium import webdriver
import pandas as pd


def locate(gene_symbols, folder):
    """
    :param folder: directory to save images
    :param gene_symbols:list of gene symbols
    :return: none
    """
    driver = webdriver.Firefox(executable_path='C:/Users/Zuohan Zhao/geckodriver.exe')
    driver.implicitly_wait(10)
    for gene in gene_symbols:
        driver.get('http://www.flyexpress.net/search.php?type=image&search=' + gene)
        images = driver.find_elements_by_xpath('//td[@class="center"]//a[@data-title="View Large Image."]')
        if len(images) == 0:
            continue
        ids = driver.find_elements_by_xpath('//td//span[3]')
        stages = driver.find_elements_by_xpath('//td//a[@data-title="Reference: Protocol"]/following-sibling::span[1]')
        gene = driver.find_element_by_xpath('//div[@id="resultWrapper"]//a[@id="gene-name"]').text
        gene_path = os.path.join(folder, gene)
        if not os.path.exists(gene_path):
            os.mkdir(gene_path)
        i = 0
        for stage, ind, image in zip(stages, ids, images):
            i += 1
            stage_path = os.path.join(gene_path, stage.text)
            if not os.path.exists(stage_path):
                os.mkdir(stage_path)
            urlretrieve(image.get_attribute('href'), os.path.join(stage_path, ind.text + '.bmp'))
            sleep(1)
            if i == 15:
                driver.find_element_by_class_name('page-forward').click()
                i = 0
                sleep(2)

if __name__ == '__main__':
    # Load gene symbols.
    gene_table = pd.read_table('../../../../Downloads/fb_synonym_fb_2020_04.tsv', skiprows=6, dtype=str, header=None)
    gene_ids = gene_table[0]
    gene_ids = gene_ids[gene_ids.str.contains('gn')]
    # Download data.
    folder = 'C:/Users/Zuohan Zhao/Pictures/test_crawl'
    if not os.path.exists(folder):
        os.mkdir(folder)
    locate(gene_ids, folder)