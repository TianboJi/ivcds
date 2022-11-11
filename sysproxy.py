import os
class SysProxy:
    def __init__(self,http="8999",https="8999",ftp="8999",socks="1089"):
        self.http = http
        self.https = https
        self.ftp = ftp
        self.socks = socks
    
    def show_proxy(self):
        http_proxy = os.environ.get('HTTP_PROXY',"")
        https_proxy = os.environ.get('HTTPS_PROXY',"")
        ftp_proxy = os.environ.get('FTP_PROXY',"")
        all_proxy = os.environ.get('ALL_PROXY',"")
        print(f'HTTP_PROXY="{http_proxy}"')
        print(f'HTTPS_PROXY="{https_proxy}"')
        print(f'FTP_PROXY="{ftp_proxy}"')
        print(f'all_PROXY="{all_proxy}"')
        
    def set_proxy(self,http=None,https=None,ftp=None,socks=None):
        if http is not None:
            self.http = http
        if https is not None:
            self.https = https
        if ftp is not None:
            self.ftp = ftp
        if socks is not None:
            self.socks = socks
        os.environ['HTTP_PROXY']=f"http://127.0.0.1:{self.http}"
        os.environ['HTTPS_PROXY']=f"http://127.0.0.1:{self.https}"
        os.environ['FTP_PROXY']=f"http://127.0.0.1:{self.ftp}"
        os.environ['ALL_PROXY']=f"socks5://127.0.0.1:{self.socks}"
        self.show_proxy()
        
    def reset_proxy(self):
        os.environ['HTTP_PROXY']=""
        os.environ['HTTPS_PROXY']=""
        os.environ['FTP_PROXY']=""
        os.environ['ALL_PROXY']=""
        self.show_proxy()
            
        
        