package com.sentinel.app

import android.content.Context
import androidx.room.Database
import androidx.room.Room
import androidx.room.RoomDatabase
import androidx.room.migration.Migration
import androidx.sqlite.db.SupportSQLiteDatabase

// The phone's own copy of every scanned message. Nothing here leaves the phone:
// the app has no pairing, no upload path and no INTERNET permission.

@Database(entities = [StoredMessage::class], version = 3, exportSchema = false)
abstract class ScanDb : RoomDatabase() {
    abstract fun messages(): MessageDao

    companion object {
        @Volatile private var instance: ScanDb? = null

        // v3 dropped the upload queue along with pairing. A real migration, not
        // a destructive one, so the stored messages and their verdicts survive.
        private val DROP_UPLOAD_QUEUE = object : Migration(2, 3) {
            override fun migrate(db: SupportSQLiteDatabase) {
                db.execSQL("DROP TABLE IF EXISTS queued_scans")
            }
        }

        fun get(context: Context): ScanDb = instance ?: synchronized(this) {
            instance ?: Room
                .databaseBuilder(context.applicationContext, ScanDb::class.java, "sentinel.db")
                .addMigrations(DROP_UPLOAD_QUEUE)
                // ponytail: only v1 (test devices, before messages were stored)
                // has no path; it is rebuilt.
                .fallbackToDestructiveMigrationFrom(1)
                .build()
                .also { instance = it }
        }
    }
}

