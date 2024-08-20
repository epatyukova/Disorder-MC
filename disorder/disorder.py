import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from pymatgen.core.composition import Composition
from disorder.cifreader import Read_CIF
import networkx as nx
from pymatgen.core.lattice import Lattice


class Disorder: 
    """ 
    Disorder class aims to assign disorder labels to each orbit in the structure spacified by the input CIF
    (1) cutoff specify how the threshold for sites intersection is determined
    (2) occ_tol determines the threshold of toleration of occupancy > 1. If occ_tol >= occupancy > 1, 
    then occupancy is rescaled to 1
    (3) merge_tol determines the threshold for the distance between the sites to be assumed 
    having the same location. In this case, they are merged, which generally leads to sites occupied 
    by more than one element
    (4) pymatgen_dist_matrix specifies which distance matrix to use. If True, than Structure.distance_matrix
    from pymatgen is used. Otherwise, the hand-written one is used.
    """
    def __init__(self, file: str, radius_file: str='data/all_radii.csv', cutoff: float=0.5,\
                 occ_tol: float=1.05, merge_tol: float=0.01, pymatgen_dist_matrix: bool=False,\
                 dist_tol: float=0.01):
        
        self.radius=pd.read_csv(radius_file,dtype={
            'symbol': str,
            'charge': int,
            'CN': str,
            'Spin State': str,
            'crystal radius': float,
            'ionic radius': float,
            'Key*': str,
            'species': str,
            'covalent': float,
            'empirical': float,
            'metallic': float
        })
        self.cutoff=cutoff
        self.occ_tol=occ_tol
        self.pymatgen_dist_matrix=pymatgen_dist_matrix
        self.errors=[]
        self.list_el=list(set(self.radius['symbol'].values))
        self.merge_tol=merge_tol
        self.dist_tol=dist_tol
    
        self.material=Read_CIF(file=file)
        
        try:
            o=self.material.orbits()
            s=self.material.symmetry()
            
            self.positions,self.struct = self.material.positions(o,s,pystruct=True,\
                                         merge_sites=True,merge_tol=merge_tol,dist_tol=dist_tol)
            
            self.errors=self.material.return_errors()
        except: 
            self.errors.append('IStructure_from_CIF not reading CIF')
    
                
    def return_errors(self):
        return self.errors
    
    def vacancy_number(self, orbit_label):
        db=self.positions.loc[self.positions['atom_site_label']==orbit_label]
        vac_number=sum(db['atom_site_occupancy'].values)
        return vac_number
    
    def create_cell(self):
        """This method creates a cell as input format in spglib"""
        lattice=self.struct.lattice.as_dict()['matrix']
        positions=[]
        elements=[]
        for s in self.struct.sites:
            positions.append(list(s.frac_coords))
            if(len(s.as_dict()['species'])==1):
                elements.append(s.as_dict()['species'][0]['element'])
            else:
                el=[]
                for x in s.as_dict()['species']:
                    el.append(x['element'])
                el.sort()
                elements.append(el)
        
        num_el=[]
        ext={}
        counter=100
        for el in elements:
            if(type(el)==str and el!='D'):
                num_el.append(self.dict_el[el])
            elif(type(el)==str and el=='D'):
                num_el.append(99)
                self.error.append('D in the structure')
            else:
                new_el=el[0]
                for e in el[1:]:
                    new_el=new_el+e
                if(new_el in ext.keys()):
                    num_el.append(ext[new_el])
                else:
                    num_el.append(counter)
                    ext[new_el]=counter
                    counter+=1
        self.dict_el.update(ext)
        return (lattice,positions,num_el)
  
    def determine_intersections(self):
        """This function determines the intersections between sites.
        Returns a list of list of sites which intersect with a given one.
        """
        num_positions = len(self.positions)
        site_element_radiuses = self.element_radiuses()

        # Create the distance template matrix
        distance_template=np.zeros((num_positions,num_positions))
        for i in range(len(self.positions)):
            for j in range(len(self.positions)):
                if(i!=j):
                    distance_template[i,j]=max([1,(site_element_radiuses[i]+site_element_radiuses[j])*self.cutoff])

        # Determine the distance matrix
        dist = self.struct.distance_matrix if self.pymatgen_dist_matrix else self.distance_matrix()

        # Calculate the difference matrix
        diff_matrix = dist - distance_template

        # Determine close sites using list comprehensions
        close_sites = [
            [j for j in range(num_positions) if diff_matrix[i, j] < 0]
            for i in range(num_positions)
        ]

        # Exclude positional disorder for 'H1+' and 'D1+' if needed
        # Uncomment if needed
        # for i, cl in enumerate(close_sites):
        #     for sp in Composition(self.positions.iloc[i]['atom_site_type_symbol']).elements:
        #         if str(sp.element) in ['H', 'D'] and sp.oxi_state == 1:
        #             close_sites[i] = []

        self.positions['intersecting_sites'] = close_sites

        return self.positions

    
    def determine_orbits(self):
        """ This function determines the equivalent positions (orbits) from
            the list of fractional coordinates"""
        
        columns=['label','species','multiplicity','Wyckoff_symbol','occupancy']
        orbits_labels=list(set(self.positions['atom_site_label']))
        arr=[]
        for orb in orbits_labels:
            dx=self.positions.loc[self.positions['atom_site_label']==orb]
            arr.append([orb,dx['atom_site_type_symbol'].values[0],dx['atom_site_symmetry_multiplicity'].values[0],
                       dx['atom_site_Wyckoff_symbol'].values[0],dx['atom_site_occupancy'].values[0]])
        
        self.orbits=pd.DataFrame(arr,columns=columns)
        
        return self.orbits
    
    def determine_intersecting_orbits(self):
        """ 
        This function determines internal and external intersections 
        for orbits based on the lists of intersections between sites 
        """

        self.orbits = self.determine_orbits()

        internal_intersection = []
        external_intersection = []
        intersect_orbits = []
        intersect_orbit_connected = []

        VPorbits = any(
            (len(orbit.species) == 1 and orbit.occupancy < 1.0) or 
            (len(orbit.species) > 1 and orbit.occupancy < 0.989)
            for orbit in self.orbits.itertuples()
            )

        if not VPorbits:
            for orbit in self.orbits.itertuples():
                internal_intersection.append(False)
                external_intersection.append(False)
                intersect_orbits.append([])
                intersect_orbit_connected.append([orbit.label])
               
            self.orbits['internal_intersection'] = internal_intersection
            self.orbits['external_intersection'] = external_intersection
            self.orbits['intersecting_orbits'] = intersect_orbits
            self.orbits['intersect_orbit_connected'] = intersect_orbit_connected

        else:
            self.positions = self.determine_intersections()

            for orbit in self.orbits.itertuples():
                db = self.positions[self.positions['atom_site_label'] == orbit.label]
                ib = set()
                internal = False

                for intersecting_sites in db['intersecting_sites']:
                    for site in intersecting_sites:
                        if site not in db.index:
                            ib.add(self.positions.at[site, 'atom_site_label'])
                        else:
                            internal = True

                internal_intersection.append(internal)
                external_intersection.append(bool(ib))
                intersect_orbits.append(list(ib))

            self.orbits['internal_intersection'] = internal_intersection
            self.orbits['external_intersection'] = external_intersection
            self.orbits['intersecting_orbits'] = intersect_orbits

            orbit_graph = nx.Graph()
            for orbit, ib in zip(self.orbits['label'], intersect_orbits):
                orbit_graph.add_node(orbit)
                for intersecting_orbit in ib:
                    orbit_graph.add_edge(orbit, intersecting_orbit)

            connected_components = [list(comp) for comp in nx.connected_components(orbit_graph)]
            for orbit in self.orbits['label']:
                for comp in connected_components:
                    if orbit in comp:
                        intersect_orbit_connected.append(comp)

            self.orbits['intersect_orbit_connected'] = intersect_orbit_connected

        return self.orbits, VPorbits
    
    def element_radiuses(self):
        """Crystal radiuses are given only for integer oxidation numbers, but
           sometimes CIF files contain float oxidations. This function rounds 
           oxidation states to use later in radius lookup table
           
           If element has oxidation state not present in the table of ionic radii, or
           its oxidation is 0, empirical radius is taken""" 
        
        element_radiuses=[]
        
        for el in self.positions['atom_site_type_symbol'].values:
            rad=[]
            species=Composition(el).elements
            # if several elements occupy the site, the average of the radiuses is taken
            # if oxidation state is non-integer, it is rounded to the smaller side to get the crystal radius
            for spec in species:
                elem=spec.element
                oxi=int(round(spec.oxi_state,0))
                if str(elem) in self.list_el:
                    rs=self.radius.loc[self.radius['symbol']==str(elem)].loc[self.radius['charge']==oxi]['ionic radius'].values
                    if(len(rs)>0):
                        rad.append(min(rs))
                    else:
                        rs=min(self.radius.loc[self.radius['symbol']==str(elem)]['empirical'].values)
                        rad.append(rs)
                else:
                    self.errors.append('Element '+str(elem)+' does not have radius data')
                    rad=np.nan
            
            element_radiuses.append(np.mean(rad))
        
        self.positions['radiuses']=element_radiuses
        return element_radiuses
    
    def distance(self, x1, x2, lattice, space_num):
        if space_num < 16 or (142 < space_num < 195):
            x1c = np.dot(x1, lattice)
            x2c = np.dot(x2, lattice)
            lattice_vectors = [0, 1, -1]
            
            # Generate image shifts
            shifts = np.array(np.meshgrid(*[lattice_vectors]*3)).T.reshape(-1, 3)
            x1_images = x1c + np.dot(shifts, lattice)
            x2_images = x2c + np.dot(shifts, lattice)

            d = cdist(x1_images, x2_images, 'euclidean')
            dist = np.min(d)
        else:
            d = np.zeros(3)
            for i in range(3):
                diff = x1[i] - x2[i]
                if diff > 0.5:
                    d[i] = diff - 1.0
                elif diff < -0.5:
                    d[i] = diff + 1.0
                else:
                    d[i] = diff
            dc = np.dot(d, lattice)
            dist = np.sqrt(np.sum(dc ** 2))
        return dist

    def distance_matrix(self):
        space_num=self.material.space_group
        lattice=Lattice.from_parameters(a=self.material.cell['abc'][0],
                                        b=self.material.cell['abc'][1],
                                        c=self.material.cell['abc'][2],
                                        alpha=self.material.cell['angles'][0],
                                        beta=self.material.cell['angles'][1],
                                        gamma=self.material.cell['angles'][2])
        lattice=lattice.as_dict()['matrix']
        # the code assumes that self.positions and self.struct are ordered in the same way
        dist_matrix=np.zeros((len(self.positions),len(self.positions)))
        if(space_num < 16 or (space_num > 142 and space_num < 195)):
            for i in range(len(self.positions)):
                for j in range(i+1,len(self.positions)):
                    x1=np.array([self.positions.iloc[i]['atom_site_fract_x'],\
                        self.positions.iloc[i]['atom_site_fract_y'],\
                        self.positions.iloc[i]['atom_site_fract_z']])
                    x2=np.array([self.positions.iloc[j]['atom_site_fract_x'],\
                        self.positions.iloc[j]['atom_site_fract_y'],\
                        self.positions.iloc[j]['atom_site_fract_z']])

                    dist_matrix[i,j]=self.distance(x1,x2,lattice,space_num)
                    dist_matrix[j,i]=dist_matrix[i,j]
        else:
            dist_matrix=self.struct.distance_matrix
                    
        return dist_matrix
    
    def classify(self):
        self.orbits, _ = self.determine_intersecting_orbits()
        orbit_disorder = []
        
        for i, orbit in self.orbits.iterrows():
            species_count = len(orbit['species'].keys())
            occupancy = orbit['occupancy']
            external_intersect = orbit['external_intersection']
            internal_intersect = orbit['internal_intersection']
            intersect_orbit_connected = orbit['intersect_orbit_connected']
            
            if not external_intersect and not internal_intersect:
                if species_count == 1:
                    orbit_disorder.append('O' if occupancy == 1.0 else 'V')
                elif occupancy > 0.989:
                    orbit_disorder.append('S')
                else:
                    orbit_disorder.append('SV')
            elif internal_intersect and not external_intersect:
                if species_count > 1:
                    orbit_disorder.append('SVP')
                else:
                    orbit_disorder.append('VP')
            else:
                intersect_species = set()
                for intersect_orb in intersect_orbit_connected:
                    intersect_species.update(self.orbits.loc[self.orbits['label'] == intersect_orb, 'species'].values[0].keys())
                
                if orbit['species'].keys() == intersect_species:
                    if species_count > 1:
                        orbit_disorder.append('SVP')
                    else:
                        orbit_disorder.append('VP')
                else:
                    orbit_disorder.append('SVP')
                        
        self.orbits['orbit_disorder'] = orbit_disorder
        return self.orbits
    
    def print_error(self):
        print(self.error)
        return
    
    def float_brackets(self,s):
        s_new=''
        for sym in s:
            if(sym=='(' or sym==')' or sym=="'"):
                pass
            else:
                s_new+=sym
        return float(s_new)
    