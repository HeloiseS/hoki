from hoki import load
import pkg_resources

data_path = pkg_resources.resource_filename('hoki', 'data')

#DATLOC = "/home/heloise/bpass_v2.2.1_imf_chab100/"

sn_file = data_path+'/supernova-bin-imf_chab100.z008.dat'
nmbr_file = data_path+'/numbers-bin-imf_chab100.z001.dat'


def test_load_sn_rates():
    data = load.sn_rates(sn_file)
    assert data.shape[0] > 0, "The DataFrame is empty"
    assert data.shape[1] == 18, "There should be 18 columns, instead there are "+str(data.shape[1])


def test_load_stellar_numbers():
    data = load.stellar_numbers(nmbr_file)
    assert data.shape[0] > 0, "The DataFrame is empty"
    assert data.shape[1] == 21, "There should be 21 columns, instead there are "+str(data.shape[1])
