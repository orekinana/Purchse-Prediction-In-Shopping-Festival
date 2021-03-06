MODEL_CONFIGS = {
    "jd":
    {
        # generation parameter
        'order_feature':30, 
        'support_feature':30, 
        'region_feature':13, 
        'G_input':10, 
        'G_hidden':[10,20], 
        'G_output':30, 
        'sample_L':10, 
        'time_num':3,

        # temporal parameter
        'embedding_out':20, 
        'sub_seq_in':20, 
        'sub_seq_out':10, 
        'mlp_in_t':10, 
        'mlp_out_t':20, 
        'tem_att_in':20, 
        'seq_in':20,
        'seq_out':10, 
        'fea_att_in':20, 
        'linear_out':20,
        'fin_in_t':20, 
        'fin_out_t':10, 
        
        # spatial parameter
        'embedding_out_tar':20, 
        'region_fea_list':[23,27,4], 
        'mlp_in_s':3, # len(region_fea_list)
        'mlp_out_s':10,
        'att_r':10, 
        'fc_out':20, 
        'fin_in_s':10, 
        'fin_out_s':10, 
        'region_num':15,
    },
}