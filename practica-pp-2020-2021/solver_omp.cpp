#include <stdio.h>
#include <math.h>
#include <omp.h>
#include <sys/time.h>
#include "wtime.h"
#include "definitions.h"
#include "energy_struct.h"


/**
* Funcion que implementa la solvatacion en openmp 
*/
extern void forces_OMP_AU (int atoms_r, int atoms_l, int nlig, float *rec_x, float *rec_y, float *rec_z, float *lig_x, float *lig_y, float *lig_z, int* rectype, int* ligtype, float *ql ,float *qr, float *energy, struct autodock_param_t *a_params, int nconformations){


	float dist, temp_desolv = 0,miatomo[3], e_desolv;
	int j,i;
	int ind1, ind2;
	int total;

	float difx,dify,difz, solv_asp_1, solv_asp_2, solv_vol_1, solv_vol_2,solv_qasp_1,solv_qasp_2;
	float  mod2x, mod2y, mod2z;	
 total = nconformations * nlig;
 omp_set_num_threads(32);
 #pragma omp parallel for private (solv_asp_1,solv_vol_2,solv_asp_2,solv_vol_1,difx,miatomo) reduction (+:temp_desolv)
	

	for (int k=0; k < (nconformations*nlig); k+=nlig)
	{
		for(int i=0;i<atoms_l;i++){					
			e_desolv = 0;
			ind1 = ligtype[i];
			miatomo[0] = *(lig_x + k + i);
			miatomo[1] = *(lig_y + k + i);
			miatomo[2] = *(lig_z + k + i);
			solv_asp_1 = a_params[ind1].asp;
			solv_vol_1 = a_params[ind1].vol;
			for(int j=0;j<atoms_r;j++){				
				e_desolv = 0;
				ind2 = rectype[j];
				solv_asp_2 = a_params[ind2].asp;
				solv_vol_2 = a_params[ind2].vol;
				difx= (rec_x[j]) - miatomo[0];
				dify= (rec_y[j]) - miatomo[1];
				difz= (rec_z[j]) - miatomo[2];
				mod2x=difx*difx;
				mod2y=dify*dify;
				mod2z=difz*difz;
			
				difx=mod2x+mod2y+mod2z;
				dist = sqrtf(difx);
							
				e_desolv = ((solv_asp_1 * solv_vol_2) + (QASP * fabs(ql[i]) *  solv_vol_2) + (solv_asp_2 * solv_vol_1) + (QASP * fabs(qr[j]) * solv_vol_1)) * exp(-difx/(2*G_D_2));
				temp_desolv += e_desolv;	
			
			}
		}
		energy[k/nlig] = temp_desolv;
		temp_desolv = 0;
	}
	printf("Desolvation term value: %f\n",energy[0]);
	

}



