import numpy as np
import vtktools
import os 
from tqdm import tqdm

def get_clean_vtu(filename):
    "Removes fields and arrays from a vtk file, leaving the coordinates/connectivity information."
    vtu_data = vtktools.vtu(filename)
    clean_vtu = vtktools.vtu()
    clean_vtu.ugrid.DeepCopy(vtu_data.ugrid)
    fieldNames = clean_vtu.GetFieldNames()
# remove all fields and arrays from this vtu
    for field in fieldNames:
        clean_vtu.RemoveField(field)
        fieldNames = clean_vtu.GetFieldNames()
        vtkdata=clean_vtu.ugrid.GetCellData()
        arrayNames = [vtkdata.GetArrayName(i) for i in range(vtkdata.GetNumberOfArrays())]
    for array in arrayNames:
        vtkdata.RemoveArray(array)
    return clean_vtu

for i in tqdm(range(100)):
    #cwd = os.getcwd()
    # if not os.path.isdir('reconstructed_vtu'):
    #     os.mkdir('reconstructed_vtu')  
    #os.chdir('csv_data') # will overwrite files in results

    # csv_data = np.loadtxt(f'/lcrc/project/SFC_Transformers/SFC-CAE/csv_data/data_{i}.csv', delimiter=',')[:,2:]
    csv_data = np.loadtxt(f'output/data_{i}.csv', delimiter=',')[:,2:]
    
    print('shape csv_data', csv_data.shape)
    csv_data = csv_data.reshape(-1, 20550*2, order='F')

    # nExamples = csv_data.shape[0]
    nNodes = 20550
    nDim = 2 # physical dimension
    print ('nNodes*nDim',nNodes*nDim)
    assert csv_data.shape[1]-nNodes*nDim == 0, "results was not the shape you were expecting"

    # get clean vtu file - path to original vtu data
    #path = '/home/cheaney/Results/nirom_test_fpc_nPOD_20_nSnap_100/snapshots/'
    filename = '/lcrc/project/SFC_Transformers/SFC-CAE/data/FPC_Re3900_DG_old/Flowpast_2d_Re3900_0.vtu'
    clean_vtu = get_clean_vtu(filename)
    velocity = np.zeros((nNodes,2)) # whether 2D or 3D

    # for i in range(nExamples):
    new_vtu = vtktools.vtu()
    new_vtu.ugrid.DeepCopy(clean_vtu.ugrid)
    new_vtu.filename = 'output_vtu/data_' + str(i) + '.vtu'
    velocity[:,0:nDim] = csv_data[0,:].reshape((nNodes,nDim),order='F')
    new_vtu.AddField('Velocity_CAE',velocity)
    new_vtu.Write()