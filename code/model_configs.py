MODEL_CONFIGS = {
    "jd":
    {
        # generation parameter
        'order_feature':30, 
        'support_feature':30, 
        'region_feature':13, 
        'G_input':5, 
        'G_hidden':[10,20], 
        'G_output':30, 
        'sample_L':10, 
        'time_num':3,

        # temporal parameter
        'embedding_out':20, 
        'sub_seq_in':20, 
        'sub_seq_out':10, 
        'mlp_in_t':10, 
        'mlp_out_t':5, 
        'tem_att_in':5, 
        'seq_in':5,
        'seq_out':10, 
        'fea_att_in':10, 
        'fin_in_t':10, 
        'fin_out_t':5,

        # spatial parameter
        'embedding_out_tar':20, 
        'region_fea_list':[23,27,4], 
        'mlp_in_s':3, # len(region_fea_list)
        'mlp_out_s':5,
        'att_r':5, 
        'fc_out':20, 
        'fin_in_s':5, 
        'fin_out_s':5, 
        'region_num':15,
    },
}