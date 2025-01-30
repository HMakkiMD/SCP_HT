PATH_FRAGMENTS='/PATH_TO_MONOMER_FILES/'
PATH_OUTPUT='/PATH_TO_POLYMERS_INPUT_STRUCTURES/'
PATH_QM='PATH_TO_QM_RESULTS'

FIVERINGS=['T','TO','TCB','TCE', 'TF', 'TFF','TCN','TCNCN','FU']
SIXRINGCONNECT=['BTZ','IG','AIG','BFF','FL','BT','BTF','BTFF','BPD','BPDF','BPDFF','BDO','BDON']


DP=['1','10']


POLYMER_DIC={'AIG(b12_2_8x2)_BT':	 ['AIG','BT'],
 'AIG(b12_2_8x2)_BTF':	 ['AIG','BTF'],
 'AIG(b12_2_8x2)_BTFF':	 ['AIG','BTFF'],
 'AIG(b12_2_8x2)_DTP(a6)':	 ['AIG','DTP'],
 'AIG(b12_2_8x2)_FU':	 ['AIG','FU'],
 'AIG(b12_2_8x2)_T':	 ['AIG','T'],
 'AIG(b12_2_8x2)_TFF':	 ['AIG','TFF'],
 'BDON(b22_4_18x2)_BT':	 ['BDON','BT'],
 'BDON(b22_4_18x2)_BTF':	 ['BDON','BTF'],
 'BDON(b22_4_18x2)_BTFF':	 ['BDON','BTFF'],
 'BDON(b22_4_18x2)_DTP(a6)':	 ['BDON','DTP'],
 'BDON(b22_4_18x2)_T':	 ['BDON','T'],
 'BDON(b22_4_18x2)_TFF':	 ['BDON','TFF'],
 'BN_DPP-b12_2_8x2_BN_TT':	 ['BN', 'DPP', 'BN', 'TT'],
 'BN_DPP-b12_2_8x2_BN_T_T':	 ['BN', 'DPP', 'BN', 'T', 'T'],
 'BN_DPP-b12_2_8x2_BN_TFF':	 ['BN', 'DPP', 'BN','TFF'],
 'BNT_DPP-b12_2_8x2_BNT_TFF':	 ['BNT', 'DPP', 'BNT', 'TFF'],
 'CPDT(a16x2)_BFF':	 ['CPDT','BFF'],
 'CPDT(b6_2_2x2)_BT':	 ['CPDT','BT'],
 'DPP-b12_2_8x2_FU_FU_FU':	['DPP','FU','FU','FU'],
 'DPP-b12_2_8x2_FU_FU':	 ['DPP','FU','FU'],
 'DPP-b12_2_8x2_FU':	 ['DPP','FU'],
 'DPP(b12_2_8x2)_T_T_T':	 ['DPP','T','T','T'],
 'DPP(b12_2_8x2)_T_T':	 ['DPP','T','T'],
 'DPP(b12_2_8x2)_T':	 ['DPP','T'],
 'FL(a8x2)_BT':	 ['FL','BT'],
 'FU_DPP-a18x2_FU_BTZ_b9_1_8':	 ['FU','DPP','FU','BTZ'],
 'FU_DPP-b12_2_8x2_FU_BFF':	 ['FU','DPP','FU','BFF'],
 'FU_DPP-b12_2_8x2_FU_TT':	 ['FU','DPP','FU','TT'],
 'FU_DPP-b12_2_8x2_FU_TVT':	 ['FU','DPP','FU','TVT'],
 'IDT(a16x4)_BO':	 ['IDT','BO'],
 'IDT(a16x4)_BPD(a1)':	 ['IDT','BPD'],
 'IDT(a16x4)_BPDF(a1)':	 ['IDT','BPDF'],
 'IDT(a16x4)_BPDFF(a1)':	 ['IDT','BPDFF'],
 'IDT(a16x4)_BT':	 ['IDT','BT'],
 'IDT(a16x4)_lBTF':	 ['IDT','BTF'],
 'IDT(a16x4)_lBTFF':	 ['IDT','BTFF'],
 'IDTT(a16x4)_BO':	 ['IDTT','BO'],
 'IDTT(a16x4)_BPD(a1)':	 ['IDTT','BPD'],
 'IDTT(a16x4)_BT':	 ['IDTT','BT'],
 'IDTT(a16x4)_BTFF':	 ['IDTT','BTF'],
 'IDTT(a16x4)_lBTF':	 ['IDTT','BTFF'],
 'IG(b12_2_8x2)_BT':	 ['IG','BT'],
 'IG(b12_2_8x2)_BTF':	 ['IG','BTF'],
 'IG(b12_2_8x2)_BTFF':	 ['IG','BTFF'],
 'IG(b12_2_8x2)_DTP(a6)':	 ['IG','DTP'],
 'IG(b12_2_8x2)_FU':	 ['IG','FU'],
 'IG(b12_2_8x2)_T':	 ['IG','T'],
 'IG(b12_2_8x2)_TFF':	 ['IG','TFF'],
 'IGF-b12_2_8x2_FU':	 ['IGF','FU'],
 'IGF(b12_2_8x2)_BT':	 ['IGF','BT'],
 'IGF(b12_2_8x2)_BTF':	 ['IGF','BTF'],
 'IGF(b12_2_8x2)_BTFF':	 ['IGF','BTFF'],
 'IGF(b12_2_8x2)_DTP(a6)':	 ['IGF','DTP'],
 'IGF(b12_2_8x2)_T':	 ['IGF','T'],
 'IGF(b12_2_8x2)_TFF':	 ['IGF','TFF'],
 'NDI-b12_2_8x2_FU_FU_FU':	 ['FU','NDI','FU','FU'],
 'NDI-b12_2_8x2_FU_FU':	 ['FU','NDI','FU'],
 'NDI-b14_2_10x2_TVT':	 ['NDI','TVT'],
 'NDI(b12_2_8x2)_T_BT_T':	['NDI','T','BT','T'],
 'NDI(b12_2_8x2)_T_BT_T':	 ['NDI','T','BT','T'],
 'NDI(b12_2_8x2)_T_T_T':	 ['T','NDI','T','T'],
 'NDI(b12_2_8x2)_T_T':	 ['T','NDI','T'],
 'NDI(b12_2_8x2)_T':	 ['NDI','T'],
 'NDTI-b12_2_8x2_FU_FU_FU':	 ['FU','NDTI','FU','FU'],
 'NDTI-b12_2_8x2_FU_FU':	 ['FU','NDTI','FU'],
 'NDTI-b12_2_8x2_FU':	 ['NDTI','FU']
 'NDTI-b14_2_10x2_BT':	 ['NDTI','BT'],
 'NDTI-b14_2_10x2_T_T_T':	 ['NDTI', 'T', 'T', 'T'],
 'NDTI-b14_2_10x2_T_T':	 ['NDTI','T','T'],
 'NDTI-b14_2_10x2_T':	 ['NDTI', 'T'],
 'T_DPP(a16x2)_T_BTZ(b9_1_8)':	 ['T','DPP','T','BTZ'],
 'T_DPP(a16x2)_T_BTZFF(b9_1_8)':	 ['T','DPP','T','BTZFF'],
 'T_DPP(b12_2_8x2)_T_BFF':	 ['T','DPP','T','BFF'],
 'T_DPP(b12_2_8x2)_T_BFFFF':	 ['T','DPP','T','BFFFF'],
 'T_DPP(b12_2_8x2)_T_T_DPP(b12_2_8x2)_T':	 ['T','DPP','T','T','DPP','T'],
 'T_DPP(b12_2_8x2)_T_TT':	 ['T','DPP','T','TT'],
 'T_DPP(b12_2_8x2)_T_TTF':	 ['T','DPP','T','TTF'],
 'T_DPP(b12_2_8x2)_T_TTFF':	 ['T','DPP','T','TTFF'],
 'T_DPP(b12_2_8x2)_T_TVT':	 ['T','DPP','T','TVT'],
 'T_DPP(b12_2_8x2)_T_TVTFF':	 ['T','DPP','T','TVTFF'],
 'TIF(a16x4)_BO':	 ['TIF','BO'],
 'TIF(a16x4)_BPD(a1)':	 ['TIF','BPD'],
 'TIF(a16x4)_BPDF(a1)':	 ['TIF','BPDF'],
 'TIF(a16x4)_BPDFF(a1)':	 ['TIF','BPDFF'],
 'TIF(a16x4)_BT':	 ['TIF','BT'],
 'TIF(a16x4)_BTF':	 ['TIF','BTF'],
 'TIF(a16x4)_BTFF':	 ['TIF','BTFF'],
 'TIG-b12_2_8x2_FU':	 ['TIG','FU'],
 'TIG(b12_2_8x2)_BT':	 ['TIG','BT'],
 'TIG(b12_2_8x2)_BTF':	 ['TIG','BTF'],
 'TIG(b12_2_8x2)_BTFF':	 ['TIG','BTF'],
 'TIG(b12_2_8x2)_DTP(a6)':	 ['TIG','DTP'],
 'TIG(b12_2_8x2)_T':	 ['TIG','T'],
 'TIG(b12_2_8x2)_TFF':	 ['TIG','TFF'],
 'TTF_DPP(b12_2_8x2)_TTF':	 ['TTF','DPP','TTF'],
 'TTFF_DPP(b12_2_8x2)_TTFF':	 ['TTFF','DPP','TTFF'],
 'TTIG-b12_2_8x2_FU':	['TTIG','FU'],
 'TTIG(b12_2_8x2)_BT':	 ['TTIG','BT'],
 'TTIG(b12_2_8x2)_BTF':	 ['TTIG','BTF'],
 'TTIG(b12_2_8x2)_BTFF':	 ['TTIG','BTFF'],
 'TTIG(b12_2_8x2)_DTP(a6)':	 ['TTIG','DTP'],
 'TTIG(b12_2_8x2)_T':	 ['TTIG','T'],
 'TTIG(b12_2_8x2)_TFF':	 ['TTIG','TFF'],
 'TT_DPP(b12_2_8x2)_TT':	 ['TT','DPP','TT']
 }
