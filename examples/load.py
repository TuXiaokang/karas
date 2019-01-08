import pickle

trainer = pickle.load(open('output/snapshot_iter_7500.pkl', 'rb'))
trainer.run()
