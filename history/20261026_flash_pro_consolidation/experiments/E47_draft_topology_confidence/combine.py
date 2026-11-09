"""Combine disjoint frozen panels by references; never duplicate coordinates."""
from .io import HERE,save,sha
from .report import report


def main():
    tags=['20261024_global_information_screen100','20261024_global_information_confirm400']
    folder=HERE/'runs/global_information_combined500';folder.mkdir(exist_ok=True)
    save(folder/'view.json',{'policies':['control','oracle_hybrid'],'parents':[{'tag':tag,'sha256':sha(HERE/'runs'/tag/'results_local.json')} for tag in tags],
        'interpretation':'privileged global information on development data; not learned model or sealed test', 'test_read':False})
    result=report(folder);assert result['variants']['control']['count']==500
    print(result['comparisons']['oracle_hybrid'])


if __name__=='__main__':main()
