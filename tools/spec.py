import os
import pprint as pp

from pyteomics import mgf

class SPEC:
    def __init__(self, meta):
        super().__init__()
        self.meta = meta

    def convert_mgf_to_spec(self, input_file, output_file=None):
        if(output_file is None):
            bn=os.path.basename(input_file)
            output_file = bn+'.spec'

        f_out=open(output_file, 'w')
        f_out.write('#Title\tPeptide\tMass\tCharge\tRTinseconds\tIons(mz:intensity)\n')

        total_peptide_num=0
        with mgf.MGF(input_file) as reader:
            for spectrum in reader:
                params = spectrum.get('params', {})
                title = params['title']
                rtinseconds = params.get('rtinseconds','-')
                rtinseconds = str(rtinseconds)

                pepmass = params['pepmass']
                pepmass = pepmass[0]
                charge = params['charge']
                charge = int(charge[0])

                precursor_mass = pepmass * charge - self.meta.proton * charge

                mz = spectrum['m/z array']
                mz = mz.astype(str)
                intensity = spectrum['intensity array']
                intensity = intensity.astype(str)
                peaks = list(zip(mz, intensity))
                peaks = [':'.join(i) for i in peaks]
                ions = ','.join(peaks)

                out_line = [title, '-', precursor_mass, charge, rtinseconds, ions]
                out_line = '\t'.join([str(i) for i in out_line])
                f_out.write(out_line+'\n')
                total_peptide_num+=1
        return(total_peptide_num)

    def convert_msp_to_spec(self, input_file, output_file=None):
        bn=os.path.basename(input_file)

        if(output_file is None):
            output_file = bn+'.spec'
        f_out=open(output_file,'w')
        prefix=bn

        #f_out.write('#Peptide\tCharge\tMW\tMods_num\tMods\tiRT\tCollision\tID\tNum_peaks\tIons(mz:intensity)')
        f_out.write('#Title\tPeptide\tMass\tCharge\tiRT\tMods_num\tMods\tCollision\tID\tNum_peaks\tIons(mz:intensity)')

        bn_arr=bn.split('_')
        collision=bn_arr[1]

        total_peptide_num=0
        flag=0
        id_n=0
        spec_id='-'
        peptide='-'
        charge = None

        f_in=open(input_file, 'r')
        for line in f_in:
            line=line.strip()
            m=re.search('^#',line)
            if(m or line==''):
                continue

            m=re.search('^Name:',line)
            if(m):
                flag=0
                num_peaks=0
                total_peptide_num+=1
                name=line.replace("Name:",'')
                name=name.strip()
                m=re.search(r'(.+)/(\d+)$',name)

                if(m):
                    peptide=m.group(1)
                    charge=m.group(2)
                    d=re.findall(r'n*\[(\d+)\]',peptide)
                    (peptide,subn)=re.subn(r'n*\[(\d+)\]','',peptide)
                    f_out.write('\n'+name+'\t'+peptide)
                    spec_id=prefix+':'+peptide+':'+str(id_n)
                    id_n+=1
                else:
                    peptide='-'
                    charge = None
                    sys.exit('Name format error! ( Name:<peptide sequence>/<charge> )')
            else:
                if(flag==1):
                    m=re.search(r'^MW:\s*(\d+(\.\d+)?)$',line)
                    if(m):
                        mw=m.group(1)
                        f_out.write('\t'+str(mw)+'\t'+str(charge))
                    else:
                        sys.exit('MW missing')
                elif(flag==2):
                    comment=line

                    m=re.search(r'\s+iRT=(\S+)$', comment)
                    if(m):
                        iRT = m.group(1)
                        f_out.write('\t'+str(iRT))
                    else:
                        sys.exit('iRT missing')
                    m=re.search(r'\s+Mods=(\d+)(/(\S+))?\s', comment)

                    if(m):
                        Mods_num = int(m.group(1))
                        if(Mods_num):
                            Mods = m.group(3)
                        else:
                            Mods='-'
                        f_out.write('\t'+str(Mods_num)+'\t'+ Mods)
                    else:
                        sys.exit('Mods missing')

                    f_out.write('\t'+collision+'\t'+spec_id)
                elif(flag==3):
                    m=re.search(r'^Num peaks:\s+(\d+)$', line)
                    if(m):
                        num_peaks=int(m.group(1))
                        f_out.write('\t'+str(num_peaks))
                elif(flag>3):
                    arr=line.split('\t')
                    mz=float(arr[0])
                    intensity=float(arr[1])
                    sep=','
                    if(flag == 4):
                        sep='\t'
                    f_out.write(sep+str(mz)+':'+str(intensity))
            flag+=1
        f_out.write('\n')

        f_in.close()
        f_out.close()
        total_peptide_num += 1
        return(total_peptide_num)
