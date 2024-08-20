import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from pymatgen.core.structure import Structure
from pymatgen.core.lattice import Lattice
from pymatgen.core.composition import Composition

from typing import Union

class Read_CIF:
    """
    A class to read cif file, list of methods:
    read_formula(self): read sum formula from cif
    read_id(self): reads collection code from cif
    read_cell(self): read cell info: a,b,c,alpha,beta,gamma
    z(self): read number of formula units Z
    space_group(self): reads space group number
    symmetry(self): reads symmetry operations from cif, returns the list of strings = lidt of symmetry operations
    orbits(self): reads orbit information from cif, returns a pandas datafram with orbit information from cif
    positions(self, orbits, symops): produces the dataframe with the list of sites from the list of orbits 
                                     and symmetry operations. Note that positions produced for substitutitonally
                                     disordered orbits are not aggregated. 
    return_errors(): returns list of errors which were encountered by the call
    add_error(): addes an arror to the list of errors

    """
    def __init__(self, file: str, occ_tol: float = 1.05):
        self.error=[]
        self.occ_tol=occ_tol
        try:
            with open(file,'r') as f:
                self.file_lines=[]
                for line in f:
                    self.file_lines.append(line)
            self.loops=[]
            for i,line in enumerate(self.file_lines):
                if(line[:5]=='loop_' or line[:14]=='#End of TTdata'):
                    self.loops.append(i)
            self.blocks={}
            for i in range(len(self.loops)-1):
                block=[]
                for line in self.file_lines[self.loops[i]:self.loops[i+1]]:
                    if(line[0]=='_'):
                        block.append(line)
                self.blocks[self.loops[i]]=block
        except:
            self.error.append('file reading failier')
    
    @property
    def read_formula(self):
        """Reading sum formula"""
        for i,line in enumerate(self.file_lines):
            if(line[:21]=='_chemical_formula_sum'):
                formula=line[23:-2]
                for j in range(i+1,len(self.file_lines)):
                    if(self.file_lines[j][:1]!="_"):
                        if(self.file_lines[j][:1]!=';'):
                            formula=formula+' '+self.file_lines[j][:-1]
                    else:
                        break
        formula=formula.replace('"','')
        formula=formula.replace("'",'')
        return formula
    
    @property
    def read_id(self):
        """Reading ICSD Collections Code"""
        for line in self.file_lines:
            if(line[:20]=='_database_code_ICSD '):
                code=int(line[20:-1])
        return code
    
    @property
    def cell(self):
        """Reading cell info. Returns lattice periods and angles"""
        switch=0
        a=0
        b=0
        c=0
        alpha=0
        beta=0
        gamma=0
        for i in range(len(self.loops[:-1])):
            for j_line in self.blocks[self.loops[i]]:
                if(j_line[:15]=='_cell_length_a '):
                    a=self._float_brackets(j_line[15:-1])
                elif(j_line[:15]=='_cell_length_b '):
                    b=self._float_brackets(j_line[15:-1])
                elif(j_line[:15]=='_cell_length_c '):
                    c=self._float_brackets(j_line[15:-1])
                elif(j_line[:18]=='_cell_angle_alpha '):
                    alpha=self._float_brackets(j_line[18:-1])
                elif(j_line[:17]=='_cell_angle_beta '):
                    beta=self._float_brackets(j_line[17:-1])
                elif(j_line[:18]=='_cell_angle_gamma '):
                    gamma=self._float_brackets(j_line[18:-1])
                switch=1
                vec_angles={}
                vec_angles['abc']=[a,b,c]
                vec_angles['angles']=[alpha,beta,gamma]
        if(switch==0):
            self.error.append('No unit cell in CIF')
            return
        else:
            return vec_angles

    @property    
    def z(self):
        switch=0
        Z=0
        for i in range(len(self.loops[:-1])):
            for j_line in self.blocks[self.loops[i]]:
                if(j_line[:22]=='_cell_formula_units_Z '):
                    switch=1
                    Z=int(j_line[22:-1])
        if(switch==0):
            self.error.append('No Z in CIF')
            return
        else:
            return Z

    @property    
    def space_group(self):
        switch=0
        space_num=0
        for i in range(len(self.loops[:-1])):
            for j_line in self.blocks[self.loops[i]]:
                if(j_line[:23]=='_space_group_IT_number '):
                    switch=1
                    space_num=int(self._float_brackets(j_line[23:-1]))
        if(switch==0):
            self.error.append('No space group number in CIF')
            return
        else:
            return space_num

   
    def symmetry(self):
        """Reading symmetry operations block"""
        switch=0
        for i in range(len(self.loops[:-1])):
            if(self.blocks[self.loops[i]]==['_space_group_symop_id\n', '_space_group_symop_operation_xyz\n']):
                switch=1
                symops=list(np.zeros(self.loops[i+1]-1-(self.loops[i]+len(self.blocks[self.loops[i]]))))
                for j in range(self.loops[i]+1+len(self.blocks[self.loops[i]]),self.loops[i+1]):
                    s=self.file_lines[j]
                    s=s.replace("'",'')
                    s=s.replace(",",'')
                    s=s.replace("\n",'')
                    aa=s.split(' ')
                    symops[int(aa[0])-1]=[aa[1],aa[2],aa[3]]
        if(switch==0):
            self.error.append('No symops in CIF')
            return
        else:
            return symops
        
    def orbits(self):
        """Reading atomic unique positions with respect to symmetry operations (orbits)"""
        switch=0
        for i in range(len(self.loops[:-1])):
            if(('_atom_site_label\n' in self.blocks[self.loops[i]]) and
               ('_atom_site_type_symbol\n' in self.blocks[self.loops[i]]) and
               ('_atom_site_symmetry_multiplicity\n' in self.blocks[self.loops[i]]) and
               ('_atom_site_fract_x\n' in self.blocks[self.loops[i]]) and
               ('_atom_site_fract_y\n' in self.blocks[self.loops[i]]) and
               ('_atom_site_fract_z\n' in self.blocks[self.loops[i]]) and
               ('_atom_site_occupancy\n' in self.blocks[self.loops[i]]) 
               ):
                switch=1
                columns=[]
                for j in range(len(self.blocks[self.loops[i]])):
                    columns.append(self.blocks[self.loops[i]][j][1:-1])
                row_list=[]
                for j in range(self.loops[i]+1+len(self.blocks[self.loops[i]]),self.loops[i+1]):
                    s=self.file_lines[j][:-1]
                    data=s.split(' ')
                    row={}
                    if(len(data)==len(columns)):
                        for l in range(len(data)):
                            row[columns[l]]=data[l]
                        row_list.append(row)
                    else:
                        self.error.append('Numbers of keys and values in structure differ')
                        return
                if(len(row_list)>0):
                    orbits = pd.DataFrame(row_list, columns=columns)
                    new=[]
                    for s in orbits['atom_site_fract_x'].values:
                        new.append(self._float_brackets(s))
                    orbits['atom_site_fract_x']=new
                    new=[]
                    for s in orbits['atom_site_fract_y'].values:
                        new.append(self._float_brackets(s))
                    orbits['atom_site_fract_y']=new
                    new=[]
                    for s in orbits['atom_site_fract_z'].values:
                        new.append(self._float_brackets(s))
                    orbits['atom_site_fract_z']=new
                    new=[]
                    for s in orbits['atom_site_symmetry_multiplicity'].values:
                        new.append(int(s))
                    orbits['atom_site_symmetry_multiplicity']=new
                    new=[]
                    for s in orbits['atom_site_occupancy'].values:
                        new.append(self._float_brackets(s))
                        if(self._float_brackets(s)>self.occ_tol):
                            self.error.append('Occupancies outside the tolerance')
                    orbits['atom_site_occupancy']=new
        if(switch==0):
            self.error.append('No structure in CIF')
            return
        else:
            return orbits
        

    def positions(self, orbits: Union[pd.DataFrame, None],symops: list,pystruct: bool=False,
                  merge_sites: bool=False, merge_tol: float=1e-2,r: int=3,dist_tol: float=1e-2):
        """
        Function creating positions from orbits and symmetry operations
        Input: orbits - pandas dataframe describing orbits, reproduces the information in CIF 
               (columns=['atom_site_label','atom_site_type_symbol','atom_site_symmetry_multiplicity',
               'atom_site_Wyckoff_symbol','atom_site_fract_x','atom_site_fract_y','atom_site_fract_z',
               'atom_site_B_iso_or_equiv', 'atom_site_occupancy'])
               output of self.orbit(). 
               symops - list[str] list of symmetry operations from CIF
        """
        
        def pbc(n):
            if(n==-0.0):
                n=0.0
            elif(n<0):
                n=n+1.0
            elif(n>=1.0):
                n=n-1.0
            return n
        
        def distance(x1,x2,lattice,space_num):
            if(space_num < 16 or (space_num > 142 and space_num < 195)):
                x1c=np.dot(x1,lattice)
                x2c=np.dot(x2,lattice)
                x1images=np.zeros((27,3))
                x2images=np.zeros((27,3))

                for i in range(3):
                    vec=np.array([[0, lattice[0][i], -lattice[0][i], lattice[1][i], -lattice[1][i],lattice[2][i],-lattice[2][i],\
                     lattice[0][i]+lattice[1][i],-lattice[0][i]-lattice[1][i],lattice[0][i]-lattice[1][i],-lattice[0][i]+lattice[1][i],\
                     lattice[2][i]+lattice[1][i],-lattice[2][i]-lattice[1][i],lattice[2][i]-lattice[1][i],-lattice[2][i]+lattice[1][i],\
                     lattice[0][i]+lattice[2][i],-lattice[0][i]-lattice[2][i],lattice[0][i]-lattice[2][i],-lattice[0][i]+lattice[2][i],\
                     lattice[0][i]+lattice[1][i]+lattice[2][i],-lattice[0][i]-lattice[1][i]-lattice[2][i],\
                     -lattice[0][i]+lattice[1][i]+lattice[2][i],lattice[0][i]-lattice[1][i]+lattice[2][i],lattice[0][i]+lattice[1][i]-lattice[2][i],\
                     -lattice[0][i]-lattice[1][i]+lattice[2][i],-lattice[0][i]+lattice[1][i]-lattice[2][i],lattice[0][i]-lattice[1][i]-lattice[2][i]]])

                    x1images[:,i]=x1c[i]*np.ones(27)+vec
                    x2images[:,i]=x2c[i]*np.ones(27)+vec

                d=cdist(x1images,x2images,'euclidean')
                dist=np.min(d)

            else:
                d=np.zeros(3)
                for i in range(3):
                    if(x1[i]-x2[i]>0.5):
                        d[i]=(x1[i]-x2[i]-1.0)
                    elif(x1[i]-x2[i]<-0.5):
                        d[i]=(x1[i]-x2[i]+1.0)
                    else:
                        d[i]=x1[i]-x2[i]
                dc=np.dot(d,lattice)
                dist=np.sqrt(dc[0]**2+dc[1]**2+dc[2]**2)

            return dist         

        def check_present(new,new_sites,lattice,space_num,tol=dist_tol):
            switch=False
            for xyz in new_sites:
                truth_string=[]
                for i in [0,1,2,3,7]:
                    if(new[i]==xyz[i]):
                        truth_string.append(True)
                    else:
                        truth_string.append(False)
                r2=distance([new[4],new[5],new[6]],[xyz[4],xyz[5],xyz[6]],lattice,space_num)
                if(r2<tol):
                    truth_string.append(True)
                else:
                    truth_string.append(False)
                if(set(truth_string)=={True}):
                    switch=True
            return switch
        
        def check_present_x(new,new_sites,lattice,space_num,tol=dist_tol):
            switch=False
            for xyz in new_sites:
                # need to calculate distance in cartesian coord + taking into account periodic 
                # boundary conditions
                r2=distance(new,xyz,lattice,space_num)
                if(r2<tol):
                    switch=True
            return switch
        
        def correct_rounding(orb,lattice,space_num,r=r,tol=dist_tol):
            columns=orb.columns.values
            site_label=orb.iloc[0]['atom_site_label']
            site_type=orb.iloc[0]['atom_site_type_symbol']
            multi=orb.iloc[0]['atom_site_symmetry_multiplicity']
            Wyckoff=orb.iloc[0]['atom_site_Wyckoff_symbol']
            occup=orb.iloc[0]['atom_site_occupancy']
            
            new_coord=[]
            for i in range(len(orb)):
                coord=np.zeros(3)
                coord[0]=pbc(round(orb.iloc[i]['atom_site_fract_x'],r))
                coord[1]=pbc(round(orb.iloc[i]['atom_site_fract_y'],r))
                coord[2]=pbc(round(orb.iloc[i]['atom_site_fract_z'],r))
                
                new=[site_label,site_type,multi,Wyckoff,coord[0],coord[1],coord[2],occup]
                if(not check_present(new,new_coord,lattice,space_num,tol=tol)):
                    new_coord.append(new)
           
            new_orb = pd.DataFrame(new_coord, columns=columns) 
            return new_orb
        
        columns=['atom_site_label','atom_site_type_symbol','atom_site_symmetry_multiplicity','atom_site_Wyckoff_symbol',
             'atom_site_fract_x','atom_site_fract_y','atom_site_fract_z','atom_site_occupancy']
        new_sites=[]
        coords=[]
        error_orbits=[]
        
        space_num=self.space_group
        lattice=Lattice.from_parameters(a=self.cell['abc'][0],b=self.cell['abc'][1],c=self.cell['abc'][2],
                                    alpha=self.cell['angles'][0],beta=self.cell['angles'][1],gamma=self.cell['angles'][2])
        
        for i in range(len(orbits)):
            site_label=orbits.iloc[i]['atom_site_label']
            site_type=orbits.iloc[i]['atom_site_type_symbol']
            multi=orbits.iloc[i]['atom_site_symmetry_multiplicity']
            Wyckoff=orbits.iloc[i]['atom_site_Wyckoff_symbol']
            occup=round(orbits.iloc[i]['atom_site_occupancy'],3)

            x=pbc(round(orbits.iloc[i]['atom_site_fract_x'],6))
            y=pbc(round(orbits.iloc[i]['atom_site_fract_y'],6))
            z=pbc(round(orbits.iloc[i]['atom_site_fract_z'],6))
            new=[site_label,site_type,multi,Wyckoff,x,y,z,occup]
            coords.append([x,y,z])
            new_sites.append(new)
            calc_multi=1
            for sym in symops:
                new_x=pbc(round(eval(sym[0]),6))
                new_y=pbc(round(eval(sym[1]),6))
                new_z=pbc(round(eval(sym[2]),6))
                new=[site_label,site_type,multi,Wyckoff,new_x,new_y,new_z,occup]
                if(not check_present(new,new_sites,lattice.matrix,space_num,tol=dist_tol)):
                    new_sites.append(new)
                    calc_multi+=1
            if(calc_multi!=multi):
                self.error.append('Multiplicity problem, nominal = '+str(multi)+', calculated = '+str(calc_multi)+', orbit_label: '+site_label)
                error_orbits.append(site_label)
        positions = pd.DataFrame(new_sites, columns=columns)
        
        # example of file which requires this is icsd_239274.cif, icsd_094411.cif, and icsd_417840.cif
        if(len(error_orbits)>0):
            for label in error_orbits:
                orb=positions.loc[positions['atom_site_label']==label]
                ind=orb.index.values
                new_orb=correct_rounding(orb,lattice.matrix,space_num,r=3,tol=dist_tol)
                positions.drop(index=ind,inplace=True)
                positions=pd.concat([positions,new_orb])
                positions.reset_index(drop=True,inplace=True)

        if(merge_sites==False and pystruct==False):
            return positions
        elif(merge_sites==False and pystruct==True):
            species=[]
            for i,sp in enumerate(positions['atom_site_type_symbol'].values):
                if(positions['atom_site_occupancy'].values[i]<self.occ_tol and positions['atom_site_occupancy'].values[i]>1.0):
                    positions['atom_site_occupancy'].values[i]=1.0
                comp={sp:positions['atom_site_occupancy'].values[i]}
                species.append(Composition(comp))

            x=(np.array([positions['atom_site_fract_x'].values])).T
            y=(np.array([positions['atom_site_fract_y'].values])).T
            z=(np.array([positions['atom_site_fract_z'].values])).T

            coords=np.concatenate([x,y,z],axis=1)
            struct=Structure(lattice=lattice,species=species,
                                coords=coords,labels=list(positions['atom_site_label'].values))
            return positions, struct
        
        elif(merge_sites==True):
            species=[]
            coords=[]
            new_sites=[]
            for i in range(len(positions)):
                linei=positions.iloc[i]
                comp={linei['atom_site_type_symbol']:linei['atom_site_occupancy']}
                lab=linei['atom_site_label']
                occ=linei['atom_site_occupancy']
                if(i<len(positions)-1):
                    for j in range(i+1,len(positions)):
                        linej=positions.iloc[j]
                        # need to calculate distance in cartesian coord + taking into account periodic 
                        # boundary conditions
                        r2=distance([linei['atom_site_fract_x'],linei['atom_site_fract_y'],linei['atom_site_fract_z']],\
                                    [linej['atom_site_fract_x'],linej['atom_site_fract_y'],linej['atom_site_fract_z']],\
                                    lattice.matrix,space_num)
                        if(r2<merge_tol):
                            if(linej['atom_site_type_symbol'] not in comp.keys()):
                                comp[linej['atom_site_type_symbol']]=linej['atom_site_occupancy']
                            else:
                                comp[linej['atom_site_type_symbol']]+=linej['atom_site_occupancy']
                            lab+=linej['atom_site_label']
                            occ+=linej['atom_site_occupancy']
                            if(linei['atom_site_symmetry_multiplicity']!=linej['atom_site_symmetry_multiplicity']):
                                self.add_error('multiplicity of merged sites are different, labels are:'+ \
                                            linei['atom_site_label']+' and '+linej['atom_site_label'])
                            if(linei['atom_site_Wyckoff_symbol']!=linej['atom_site_Wyckoff_symbol']):
                                self.add_error('Wyckoff symbols of merged sites are different, labels are:'+ \
                                            linei['atom_site_label']+' and '+linej['atom_site_label'])
                # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                # !!!in the future need to write proper averaging of coordinates of merged sites!!!
                # !!!in cartesian coordinates with mapping back to fractional                   !!!
                # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                new=[linei['atom_site_fract_x'],linei['atom_site_fract_y'],linei['atom_site_fract_z']]
                if(not check_present_x(new,coords,lattice.matrix,space_num,tol=merge_tol)):
                    coords.append(new)
                    if(occ>1.0 and occ<=self.occ_tol):
                        self.add_error('occupancy of '+lab+' is '+str(occ)+', rescaled to 1.0')
                        for key,val in comp.items(): # rescalling occupancies in composition, otherwise
                                                # Structure will fail
                            val=round(float(val)/float(occ),4)
                            comp[key]=val
                        occ=1.0
                    if(occ>1.05):
                        self.add_error('occupancy of '+lab+' is '+str(occ))
                    species.append(Composition(comp))
                    new_sites.append([lab,comp,linei['atom_site_symmetry_multiplicity'],\
                                    linei['atom_site_Wyckoff_symbol'], new[0],new[1],new[2],occ])
            
            columns=['atom_site_label','atom_site_type_symbol','atom_site_symmetry_multiplicity','atom_site_Wyckoff_symbol',
                'atom_site_fract_x','atom_site_fract_y','atom_site_fract_z','atom_site_occupancy']
            positions = pd.DataFrame(new_sites, columns=columns)

            if(pystruct==True):
                try:
                    struct=Structure(lattice=lattice,species=species,\
                                coords=coords,labels=list(positions['atom_site_label'].values))
                except:
                    self.add_error('pymatgen Structure failed')
                    struct=None

                return positions, struct
            else:
                return positions

    def return_errors(self):
        return self.error
    
    def add_error(self,err):
        self.error.append(err)
        return
    
    def _float_brackets(self,ss):
        """
        transforms numbers from string of format x.y(z) into float x.yz
        """
        s_new=''
        for s in ss:
            if(s=='(' or s==')' or s=="'"):
                pass
            else:
                s_new+=s
        if(ss=='.' or ss=='-.'):
            s_new=0
        return float(s_new)