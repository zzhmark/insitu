import socket
from selenium import webdriver
from time import sleep


# Set web driver.
driver = webdriver.Firefox(executable_path='C:/Users/Zuohan Zhao/geckodriver.exe')
driver.implicitly_wait(30)
# Get IP.
ip = socket.gethostbyname(socket.gethostname())

# Load gene symbols.
gene_symbols = []


# Download data.
for gene in gene_symbols:
    driver.get('http://www.flyexpress.net')
    sleep(1)
    driver.find_elements_by_css_selector()

try:
    driver.get('http://skxb.seu.edu.cn/ch/author/login.aspx')
    sleep(1)
    driver.find_element_by_id('UserName').send_keys('Site-428@srtp.cn')
    driver.find_element_by_id('Password').send_keys('TCDCPP428')
    driver.find_element_by_id('Submit').click()
    sleep(2)
    driver.get('http://skxb.seu.edu.cn/mainx.aspx?psu=F5D1363D5C4A96BFF8719374AC2BFB1CECF0C9D3216FC457&login_flag=author&jid=dndxxbsk#modify_author_ui')
    sleep(1)
    driver.find_element_by_id('Menu_modify_author_ui').click()
    sleep(1)
    driver.find_element_by_id('zhicheng').clear()
    driver.find_element_by_id('zhicheng').send_keys(str(now))
    driver.find_element_by_id('weixin').clear()
    driver.find_element_by_id('weixin').send_keys(ip)
    driver.find_element_by_id('modify_author_ui_form1_btn_submit').click()
    sleep(1)
    driver.quit()
except:
    driver.quit()