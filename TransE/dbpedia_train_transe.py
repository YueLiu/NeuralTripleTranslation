import config
import models

con = config.Config()
con.set_in_path("./benchmarks/DBPEDIA/")
#True: Input test files from the same folder.
con.set_log_on(1)
con.set_work_threads(8)
con.set_train_times(1000)
con.set_nbatches(100)
con.set_alpha(0.001)
con.set_bern(0)
con.set_dimension(100)
con.set_margin(1.0)
con.set_ent_neg_rate(1)
con.set_rel_neg_rate(0)
con.set_opt_method("SGD")
#Model parameters will be exported via torch.save() automatically.
con.set_export_files("./res/dbpedia_transe.pt")
#Model parameters will be exported to json files automatically.
con.set_out_files("./res/dbpedia_embedding.vec.json")
con.init()
con.set_model(models.TransE)
con.import_variables("./res/dbpedia_transe.pt")
con.run()
