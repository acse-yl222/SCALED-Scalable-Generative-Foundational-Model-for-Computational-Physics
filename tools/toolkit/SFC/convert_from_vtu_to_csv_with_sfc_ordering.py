import os
#os.environ['KMP_DUPLICATE_LIB_OK']='True'

import space_filling_decomp_new as sfc
import sys
import vtk, vtktools
import numpy as np
import time
import shutil
from tqdm import tqdm

# calculate the space-filling curve for the mesh and convert to csv format

#----------------------------------------------------------------------------------------------
# github simple CG example
# path = 'data/rectangle/'
# filebase = 'rectangle_'
# filename = path + filebase + '0.vtu' # this file will do (all files have the same mesh)
# mesh_type = 'CG' # 'DG' or 'CG' (likely to be DG if nNodes (nPoints in Paraview) is an integer multiple (3 or 4) or nElements (nCells in Paraview))
# nTotalExamples = 1 # maximum is 1000 for this data set
# solutions_exist = False

# github CG example
#path = 'data/FPC_Re3900_CG_new/'
#filebase = 'fpc_'
#filename = path + filebase + '0.vtu' # this file will do (all files have the same mesh)
#mesh_type = 'CG' # 'DG' or 'CG' (likely to be DG if nNodes (nPoints in Paraview) is an integer multiple (3 or 4) or nElements (nCells in Paraview))
#nTotalExamples = 10 # maximum is 10 for this data set
#solutions_exist = True

# github DG example
path = 'data/SFC_data_vtu/'
filebase = 'fpc_'
filename = path + filebase + '0.vtu' #   
mesh_type = 'DG' # 'DG' or 'CG' (likely to be DG if nNodes (nPoints in Paraview) is an integer multiple (3 or 4) or nElements (nCells in Paraview))
nTotalExamples = 4000 # maximum is 10 for this data set
solutions_exist = True

# local DG example
# path = '/home/cheaney/Results/nirom_test_fpc_nPOD_20_nSnap_100/snapshots/'
# filename = path + 'Flowpast_2d_Re3900_0.vtu' # this file will do (all files have the same mesh)
# mesh_type = 'DG' # 'DG' or 'CG' (likely to be DG if nNodes (nPoints in Paraview) is an integer multiple (3 or 4) or nElements (nCells in Paraview))
# nTotalExamples = 10 # maximum is 1000 for this data set


vtu_data = vtktools.vtu(filename)
coords = vtu_data.GetLocations()
nNodes = coords.shape[0]

    
# form node connectivity
if mesh_type == 'DG':   
    findm,colm,ncolm = sfc.form_spare_matric_from_pts( coords, nNodes )
    colm = np.ones((ncolm),dtype = 'int')
    colm = colm[0:ncolm]
elif mesh_type == 'CG':
    ncolm=0
    colm=[]
    findm=[0]
    for nod in range(nNodes):
        nodes = vtu_data.GetPointPoints(nod)
        nodes2 = np.sort(nodes) #sort_assed(nodes) 
        colm.extend(nodes2[:]) 
        nlength = nodes2.shape[0]
        ncolm=ncolm+nlength
        findm.append(ncolm)

    colm = np.array(colm)
    colm = colm + 1
    findm = np.array(findm)
    findm = findm + 1

# sfc settings
ncurve = 2
graph_trim = -10  # has always been set at -10
starting_node = 0 # =0 do not specifiy a starting node, otherwise, specify the starting node

        
# call the sfc fortran code (space_filling_decomp_new.so)
print('Generating SFC...')
t0 = time.time() 
whichd, inverse_sfc_numbering = sfc.ncurve_python_subdomain_space_filling_curve(colm, findm, starting_node, graph_trim, ncurve, nNodes, ncolm)
t_sfc = time.time() - t0

N = len(inverse_sfc_numbering)
sfc_numbering = np.zeros((N, ncurve), dtype=np.int32)

# fortran numbers from 1 to N :-) whereas python numbers from 0 to N-1
inverse_sfc_numbering = inverse_sfc_numbering - 1
    
sfc_numbering[:, 0] = np.argsort(inverse_sfc_numbering[:, 0])
sfc_numbering[:, 1] = np.argsort(inverse_sfc_numbering[:, 1])
# to find the `inverse' of the sfc ordering from the sfc ordering (and vice verse)
#sfc_numbering(old_nod)=new_nod #(ond_nod is the gmsh node)
#do old_nod=1,nonods
#    new_nod = sfc_numbering(old_nod)
#    inverse_sfc_numbering(new_nod)=old_nod
#end do


for i in range(ncurve):
    print('\nsfc number',i+1)
    if N<20:
        print('sfc (Fortran convention):        ',sfc_numbering[:, i]+1)
        #print('sfc_inv (Fortran convention):    ',inverse_sfc_numbering[:, i]+1)    
        #print('sfc_inv_inv (Fortran convention):',np.argsort(inverse_sfc_numbering[:, i])+1)    
    print('min and max of sfc numbering', np.min(sfc_numbering[:,i]), np.max(sfc_numbering[:,i]) )
    print('min and max of inverse sfc numbering', np.min(inverse_sfc_numbering[:,i]), np.max(inverse_sfc_numbering[:,i]) )

print('\nTime to generate sfc (secs): ', t_sfc)

    
# read in data and save, with sfc ordering, to csv file   
t_save_to_csv = 0
t_read_in = 0

if os.path.isdir('data/SFC_data_csv'):
    shutil.rmtree('data/SFC_data_csv')
os.mkdir('data/SFC_data_csv')


print('\nWriting SFC ordering to file and solution fields to file (in gmsh ordering) ...')
for data_i in tqdm(range(nTotalExamples)):

    t0 = time.time()    
    filename = path + filebase + str(data_i) + '.vtu'
    vtu_data = vtktools.vtu(filename)
    # D[:,0] and D[:,1] store sfc #1 and sfc #2, D[:,2] and D[:,3]store velocity

    t1 = time.time()
    t_read_in = t_read_in + t1 - t0

    D = np.zeros((nNodes, 4)) # hard-wired for 2D, D = np.zeros((nNodes, 6)) for 3D

    D[:, :2] = sfc_numbering 
    
    if solutions_exist:
        velocity = vtu_data.GetField('Velocity')
        #D[:, 2] = np.sqrt(velocity[:nNodes, 0]**2 + velocity[:nNodes, 1]**2) # not needed
        D[:, 2] = velocity[:nNodes, 0] # gmsh ordering
        D[:, 3] = velocity[:nNodes, 1] # gmsh ordering

    t0 = time.time()
    #np.savetxt('csv_data/data_' + str(data_i) + '.csv', D, delimiter=',') 
    np.savetxt('data/SFC_data_csv/data_' + str(data_i) + '.csv', D, fmt='%i, %i, %26.18e, %26.18e') # hard-wired for 2D
    t1 = time.time()
    t_save_to_csv = t_save_to_csv + t1 - t0

    #if data_i%100==0:
    #    print("... data loaded and written to file #", data_i)


print('\nTime loading data (secs):    ', t_read_in)
print('Time to write to csv (secs): ', t_save_to_csv)


